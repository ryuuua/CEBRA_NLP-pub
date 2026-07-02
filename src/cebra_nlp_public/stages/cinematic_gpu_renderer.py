"""GPU cinematic frame renderer for 3D trajectory visualizations.

This module is intentionally standalone so orchestration code can feed one
frame at a time without depending on the rest of the cinematic pipeline.
It uses moderngl on an offscreen EGL context, draws circular particle sprites
with gl_PointCoord, and returns a PIL image or RGBA numpy array.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
from PIL import Image

try:  # pragma: no cover - env-dependent import
    import moderngl
except Exception:  # pragma: no cover - env-dependent import
    moderngl = None


__all__ = [
    "CameraFrame",
    "CinematicGPUFrameRenderer",
    "FrameSpec",
    "LineBatch",
    "ParticleBatch",
    "RendererOptions",
    "is_moderngl_available",
]


RGBA = tuple[float, float, float, float]
RGB = tuple[float, float, float]


def is_moderngl_available() -> bool:
    return moderngl is not None


def _as_float32_array(values: np.ndarray | Sequence[float], *, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected array with ndim={ndim}, got shape={array.shape}.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_rgba(color: Sequence[float] | str, alpha: float | None = None) -> RGBA:
    if isinstance(color, str):
        if color.startswith("#") and len(color) in {7, 9}:
            red = int(color[1:3], 16) / 255.0
            green = int(color[3:5], 16) / 255.0
            blue = int(color[5:7], 16) / 255.0
            alpha_value = int(color[7:9], 16) / 255.0 if len(color) == 9 else 1.0
            if alpha is not None:
                alpha_value = float(alpha)
            return (red, green, blue, alpha_value)
        raise ValueError(f"Unsupported color string: {color!r}")

    values = tuple(float(v) for v in color)
    if len(values) == 3:
        return (values[0], values[1], values[2], 1.0 if alpha is None else float(alpha))
    if len(values) == 4:
        return (values[0], values[1], values[2], values[3] if alpha is None else float(alpha))
    raise ValueError(f"Expected RGB or RGBA color, got {color!r}.")


def _rgba_to_u8(color: RGBA) -> tuple[int, int, int, int]:
    return tuple(int(round(max(0.0, min(1.0, channel)) * 255.0)) for channel in color)


def _resolve_internal_resolution(output_size: tuple[int, int], supersample_scale: float) -> tuple[int, int]:
    width = max(1, int(output_size[0]))
    height = max(1, int(output_size[1]))
    scale = max(1.0, float(supersample_scale))
    return (
        max(width, int(round(width * scale))),
        max(height, int(round(height * scale))),
    )


@dataclass(frozen=True, slots=True)
class ParticleBatch:
    """A uniform-colored point cloud batch.

    Coordinates are interpreted in world space and projected by the frame's MVP.
    Sizes are specified in output-pixel units and scaled automatically to the
    internal supersampled framebuffer.
    """

    positions: np.ndarray
    color: Sequence[float] | str = (1.0, 1.0, 1.0, 1.0)
    point_size_px: float = 12.0
    core_scale: float = 1.0
    halo_scale: float = 2.5
    core_alpha: float = 1.0
    halo_alpha: float = 0.35
    core_sharpness: float = 3.5
    halo_sharpness: float = 1.35

    def normalized(self) -> tuple[np.ndarray, RGBA]:
        positions = _as_float32_array(self.positions, ndim=2)
        if positions.shape[1] != 3:
            raise ValueError(f"Expected positions with shape (N, 3), got {positions.shape}.")
        return positions, _as_rgba(self.color)


@dataclass(frozen=True, slots=True)
class LineBatch:
    """A line strip batch for trails, guides, or centroids."""

    positions: np.ndarray
    color: Sequence[float] | str = (1.0, 1.0, 1.0, 1.0)
    line_width_px: float = 1.0

    def normalized(self) -> tuple[np.ndarray, RGBA]:
        positions = _as_float32_array(self.positions, ndim=2)
        if positions.shape[1] != 3:
            raise ValueError(f"Expected positions with shape (N, 3), got {positions.shape}.")
        return positions, _as_rgba(self.color)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """Single-frame camera transform."""

    mvp_matrix: np.ndarray

    def normalized_mvp(self) -> np.ndarray:
        matrix = _as_float32_array(self.mvp_matrix, ndim=2)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected 4x4 MVP matrix, got {matrix.shape}.")
        return matrix


@dataclass(frozen=True, slots=True)
class FrameSpec:
    """Renderable contents for one cinematic frame."""

    camera: CameraFrame
    particle_batches: Sequence[ParticleBatch] = field(default_factory=tuple)
    line_batches: Sequence[LineBatch] = field(default_factory=tuple)
    background_color: Sequence[float] | str = (0.02, 0.03, 0.06, 1.0)
    output_size: tuple[int, int] | None = None
    downsample_filter: int = Image.Resampling.LANCZOS


@dataclass(frozen=True, slots=True)
class RendererOptions:
    """Global renderer controls."""

    output_size: tuple[int, int] = (1920, 1080)
    supersample_scale: float = 2.0
    background_color: Sequence[float] | str = (0.02, 0.03, 0.06, 1.0)
    depth_cue_strength: float = 0.22
    depth_fog_strength: float = 0.18
    depth_fog_cool_mix: float = 0.16
    depth_fog_color: Sequence[float] | str = (0.10, 0.16, 0.28, 1.0)
    max_point_size_px: float = 128.0
    context_backend: str = "egl"
    clear_alpha: float = 1.0
    default_downsample_filter: int = Image.Resampling.LANCZOS

    def internal_size(self) -> tuple[int, int]:
        return _resolve_internal_resolution(self.output_size, self.supersample_scale)


class CinematicGPUFrameRenderer:
    """Render one cinematic 3D frame to a PIL image or RGBA array.

    The renderer owns an offscreen moderngl context and can be reused across
    many frames. Call ``close()`` or use it as a context manager.
    """

    def __init__(self, options: RendererOptions) -> None:
        if moderngl is None:
            raise RuntimeError(
                "moderngl is not available. The GPU cinematic renderer requires moderngl/EGL."
            )

        self.options = options
        self.output_size = (int(options.output_size[0]), int(options.output_size[1]))
        self.internal_size = options.internal_size()
        self._closed = False
        self._ctx = self._create_context()
        self._ctx.enable_only(moderngl.BLEND | moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._color_texture = self._ctx.texture(self.internal_size, 4)
        self._color_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._depth_buffer = self._ctx.depth_renderbuffer(self.internal_size)
        self._framebuffer = self._ctx.framebuffer(
            color_attachments=[self._color_texture],
            depth_attachment=self._depth_buffer,
        )

        self._point_program = self._ctx.program(
            vertex_shader=self._point_vertex_shader(),
            fragment_shader=self._point_fragment_shader(),
        )
        self._line_program = self._ctx.program(
            vertex_shader=self._line_vertex_shader(),
            fragment_shader=self._line_fragment_shader(),
        )

    def __enter__(self) -> "CinematicGPUFrameRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for resource in (
            getattr(self, "_framebuffer", None),
            getattr(self, "_depth_buffer", None),
            getattr(self, "_color_texture", None),
            getattr(self, "_point_program", None),
            getattr(self, "_line_program", None),
            getattr(self, "_ctx", None),
        ):
            try:
                if resource is not None:
                    resource.release()
            except Exception:
                pass

    def render_frame(self, frame: FrameSpec) -> Image.Image:
        """Render one frame and return a PIL RGB image."""

        rgba = self.render_frame_rgba(frame)
        image = Image.fromarray(rgba, mode="RGBA")
        target_size = frame.output_size or self.output_size
        if image.size != target_size:
            image = image.resize(target_size, frame.downsample_filter)
        return image.convert("RGB")

    def render_frame_rgba(self, frame: FrameSpec) -> np.ndarray:
        """Render one frame and return an RGBA uint8 numpy array."""

        self._ensure_open()

        camera_mvp = frame.camera.normalized_mvp()
        background = _as_rgba(frame.background_color, alpha=self.options.clear_alpha)
        self._framebuffer.use()
        self._ctx.clear(*background)

        for batch in frame.line_batches:
            self._draw_line_batch(batch, camera_mvp)
        for batch in frame.particle_batches:
            self._draw_particle_batch(batch, camera_mvp)

        raw = self._framebuffer.read(components=4, alignment=1)
        rgba = np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.internal_size[1], self.internal_size[0], 4)
        )[::-1, :, :]

        if self.internal_size != self.output_size:
            image = Image.fromarray(rgba, mode="RGBA")
            image = image.resize(self.output_size, self.options.default_downsample_filter)
            rgba = np.asarray(image, dtype=np.uint8)
        return np.ascontiguousarray(rgba)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("CinematicGPUFrameRenderer is closed.")

    def _create_context(self):
        backend = str(self.options.context_backend).strip().lower()
        try:
            return moderngl.create_standalone_context(backend=backend, require=330)
        except Exception as first_error:
            if backend != "egl":
                try:
                    return moderngl.create_standalone_context(backend="egl", require=330)
                except Exception:
                    pass
            try:
                return moderngl.create_standalone_context(require=330)
            except Exception as second_error:  # pragma: no cover - backend-specific failure path
                raise RuntimeError(
                    "Failed to create a moderngl standalone context for cinematic rendering."
                ) from second_error

    @staticmethod
    def _point_vertex_shader() -> str:
        return """
            #version 330
            uniform mat4 u_mvp;
            uniform float u_point_size;
            in vec3 in_pos;
            out float v_depth;
            void main() {
                vec4 clip = u_mvp * vec4(in_pos, 1.0);
                gl_Position = clip;
                gl_PointSize = u_point_size;
                float w = max(abs(clip.w), 1e-6);
                v_depth = clamp(clip.z / w * 0.5 + 0.5, 0.0, 1.0);
            }
        """

    @staticmethod
    def _point_fragment_shader() -> str:
        return """
            #version 330
            uniform vec4 u_color;
            uniform float u_core_sharpness;
            uniform float u_halo_sharpness;
            uniform float u_core_alpha;
            uniform float u_halo_alpha;
            uniform float u_depth_cue_strength;
            uniform float u_depth_fog_strength;
            uniform float u_depth_fog_cool_mix;
            uniform vec3 u_fog_color;
            in float v_depth;
            out vec4 f_color;
            void main() {
                vec2 uv = gl_PointCoord * 2.0 - 1.0;
                float r2 = dot(uv, uv);
                if (r2 > 1.0) {
                    discard;
                }

                float core = pow(max(0.0, 1.0 - r2), u_core_sharpness);
                float halo = pow(max(0.0, 1.0 - r2), u_halo_sharpness);
                float depth_t = clamp(v_depth, 0.0, 1.0);
                float depth_mix = clamp(u_depth_cue_strength * depth_t, 0.0, 1.0);
                float fog_mix = clamp(u_depth_fog_strength * depth_t, 0.0, 1.0);
                fog_mix = max(fog_mix, clamp(u_depth_fog_cool_mix * depth_t, 0.0, 1.0));

                vec3 rgb = mix(u_color.rgb, u_fog_color, fog_mix);
                float alpha = u_color.a * (u_core_alpha * core + u_halo_alpha * halo);
                alpha *= 1.0 - 0.44 * depth_mix;
                f_color = vec4(rgb, clamp(alpha, 0.0, 1.0));
            }
        """

    @staticmethod
    def _line_vertex_shader() -> str:
        return """
            #version 330
            uniform mat4 u_mvp;
            in vec3 in_pos;
            out float v_depth;
            void main() {
                vec4 clip = u_mvp * vec4(in_pos, 1.0);
                gl_Position = clip;
                float w = max(abs(clip.w), 1e-6);
                v_depth = clamp(clip.z / w * 0.5 + 0.5, 0.0, 1.0);
            }
        """

    @staticmethod
    def _line_fragment_shader() -> str:
        return """
            #version 330
            uniform vec4 u_color;
            uniform float u_depth_cue_strength;
            uniform float u_depth_fog_strength;
            uniform float u_depth_fog_cool_mix;
            uniform vec3 u_fog_color;
            in float v_depth;
            out vec4 f_color;
            void main() {
                float depth_t = clamp(v_depth, 0.0, 1.0);
                float depth_mix = clamp(u_depth_cue_strength * depth_t, 0.0, 1.0);
                float fog_mix = clamp(u_depth_fog_strength * depth_t, 0.0, 1.0);
                fog_mix = max(fog_mix, clamp(u_depth_fog_cool_mix * depth_t, 0.0, 1.0));
                vec3 rgb = mix(u_color.rgb, u_fog_color, fog_mix);
                float alpha = u_color.a * (1.0 - 0.30 * depth_mix);
                f_color = vec4(rgb, clamp(alpha, 0.0, 1.0));
            }
        """

    def _draw_particle_batch(self, batch: ParticleBatch, mvp_matrix: np.ndarray) -> None:
        positions, color = batch.normalized()
        if positions.size == 0:
            return

        point_size = float(batch.point_size_px) * max(1.0, float(self.options.supersample_scale))
        point_size = min(point_size, float(self.options.max_point_size_px))
        core_size = max(1.0, point_size * float(batch.core_scale))
        halo_size = max(core_size, point_size * float(batch.halo_scale))

        self._draw_point_pass(
            positions,
            mvp_matrix=mvp_matrix,
            color=(color[0], color[1], color[2], color[3] * float(batch.halo_alpha)),
            point_size=halo_size,
            core_sharpness=float(batch.halo_sharpness),
            halo_sharpness=max(0.8, float(batch.halo_sharpness) * 0.60),
        )
        self._draw_point_pass(
            positions,
            mvp_matrix=mvp_matrix,
            color=(color[0], color[1], color[2], color[3] * float(batch.core_alpha)),
            point_size=core_size,
            core_sharpness=float(batch.core_sharpness),
            halo_sharpness=max(0.8, float(batch.core_sharpness) * 0.52),
        )

    def _draw_point_pass(
        self,
        positions: np.ndarray,
        *,
        mvp_matrix: np.ndarray,
        color: RGBA,
        point_size: float,
        core_sharpness: float,
        halo_sharpness: float,
    ) -> None:
        vbo = self._ctx.buffer(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
        vao = self._ctx.simple_vertex_array(self._point_program, vbo, "in_pos")
        self._point_program["u_mvp"].write(np.asarray(mvp_matrix, dtype=np.float32).T.tobytes())
        self._point_program["u_point_size"].value = float(point_size)
        self._point_program["u_color"].value = tuple(float(v) for v in color)
        self._point_program["u_core_sharpness"].value = float(core_sharpness)
        self._point_program["u_halo_sharpness"].value = float(halo_sharpness)
        self._point_program["u_core_alpha"].value = 1.0
        self._point_program["u_halo_alpha"].value = 0.42
        self._point_program["u_depth_cue_strength"].value = float(self.options.depth_cue_strength)
        self._point_program["u_depth_fog_strength"].value = float(self.options.depth_fog_strength)
        self._point_program["u_depth_fog_cool_mix"].value = float(self.options.depth_fog_cool_mix)
        self._point_program["u_fog_color"].value = tuple(
            float(v) for v in _as_rgba(self.options.depth_fog_color)[:3]
        )
        self._ctx.line_width = 1.0
        vao.render(mode=moderngl.POINTS)
        vao.release()
        vbo.release()

    def _draw_line_batch(self, batch: LineBatch, mvp_matrix: np.ndarray) -> None:
        positions, color = batch.normalized()
        if positions.shape[0] < 2:
            return

        line_width = max(1.0, float(batch.line_width_px) * max(1.0, float(self.options.supersample_scale)))
        vbo = self._ctx.buffer(np.ascontiguousarray(positions, dtype=np.float32).tobytes())
        vao = self._ctx.simple_vertex_array(self._line_program, vbo, "in_pos")
        self._line_program["u_mvp"].write(np.asarray(mvp_matrix, dtype=np.float32).T.tobytes())
        self._line_program["u_color"].value = tuple(float(v) for v in color)
        self._line_program["u_depth_cue_strength"].value = float(self.options.depth_cue_strength)
        self._line_program["u_depth_fog_strength"].value = float(self.options.depth_fog_strength)
        self._line_program["u_depth_fog_cool_mix"].value = float(self.options.depth_fog_cool_mix)
        self._line_program["u_fog_color"].value = tuple(
            float(v) for v in _as_rgba(self.options.depth_fog_color)[:3]
        )
        self._ctx.line_width = line_width
        vao.render(mode=moderngl.LINE_STRIP)
        vao.release()
        vbo.release()


def render_frame_to_image(
    frame: FrameSpec,
    *,
    options: RendererOptions,
) -> Image.Image:
    """Convenience one-shot renderer."""

    with CinematicGPUFrameRenderer(options) as renderer:
        return renderer.render_frame(frame)


def render_frame_to_rgba(
    frame: FrameSpec,
    *,
    options: RendererOptions,
) -> np.ndarray:
    """Convenience one-shot renderer returning an RGBA uint8 array."""

    with CinematicGPUFrameRenderer(options) as renderer:
        return renderer.render_frame_rgba(frame)
