from manim import *
import numpy as np
import yaml
from pathlib import Path


# ============================================================
# LOAD STYLE
# ============================================================

STYLE_PATH = Path(__file__).with_name("visual-style_dark.yml")

with open(STYLE_PATH, "r", encoding="utf-8") as f:
    STYLE = yaml.safe_load(f)

config.background_color = STYLE["canvas"]["background"]


# ============================================================
# SCENE
# ============================================================

class BlogIntro3D(ThreeDScene):

    def make_beam(
        self,
        start,
        end,
        glow_color,
        core_color,
        outer_radius=0.34,
        inner_radius=0.11,
        outer_opacity=0.07,
        inner_opacity=0.92,
        resolution=(12, 12),
    ):
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)

        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length
        center = 0.5 * (start + end)

        glow = Cylinder(
            radius=outer_radius,
            height=length,
            direction=direction,
            resolution=resolution,
        )
        glow.set_fill(glow_color, opacity=outer_opacity)
        glow.set_stroke(width=0, opacity=0)

        core = Cylinder(
            radius=inner_radius,
            height=length,
            direction=direction,
            resolution=resolution,
        )
        core.set_fill(core_color, opacity=inner_opacity)
        core.set_stroke(width=0, opacity=0)

        beam = VGroup(glow, core)
        beam.move_to(center)
        return beam

    def construct(self):

        # ====================================================
        # CAMERA
        # ====================================================

        self.set_camera_orientation(
            phi=70 * DEGREES,
            theta=-48 * DEGREES,
        )
        self.renderer.camera.scale(0.60)

        # ====================================================
        # COLORS
        # ====================================================

        substrate_fill = STYLE["colors"]["foreground"]["primary"]

        pillar_fill = STYLE["surfaces"]["primary"]["fill"]
        pillar_edge = STYLE["surfaces"]["primary"]["stroke"]

        red = STYLE["colors"]["palette"]["red"]["main"]
        red_light = STYLE["colors"]["palette"]["red"]["light"]

        # ====================================================
        # VERY LARGE SUBSTRATE
        # ====================================================

        substrate = Prism(dimensions=[30.0, 22.0, 0.42])

        substrate.set_fill(
            substrate_fill,
            opacity=0.08,
        )
        substrate.set_stroke(
            substrate_fill,
            width=0.6,
            opacity=0.16,
        )
        substrate.move_to([0.0, 0.0, -0.21])

        # ====================================================
        # MANY FACETED PILLARS
        # low resolution = faceted edges
        # no stroke = no wireframe
        # ====================================================

        pillars = VGroup()

        xs = np.arange(-8.0, 8.01, 1.7)
        ys = np.arange(-6.0, 6.01, 1.7)

        base_radius = 0.34

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):

                radius = base_radius + 0.03 * np.sin(0.8 * i + 0.5 * j)
                height = 0.95 + 0.30 * np.cos(0.7 * i - 0.6 * j)

                pillar = Cylinder(
                    radius=radius,
                    height=height,
                    direction=OUT,
                    resolution=(8, 8),   # faceted silhouette
                )

                pillar.set_fill(
                    pillar_fill,
                    opacity=0.92,
                )

                # IMPORTANT:
                # tiny stroke only, so silhouette gets a hint of edge
                # without turning into wireframe
                pillar.set_stroke(
                    pillar_edge,
                    width=0.20,
                    opacity=0.10,
                )

                pillar.move_to([x, y, height / 2])

                pillars.add(pillar)

        # ====================================================
        # BEAM GEOMETRY
        # ====================================================

        hit_point = np.array([-1.1, 1.0, 0.48])

        incident_beam = self.make_beam(
            start=[-10.5, 4.4, 2.7],
            end=hit_point,
            glow_color=red,
            core_color=red_light,
            outer_radius=0.46,
            inner_radius=0.12,
            outer_opacity=0.06,
            inner_opacity=0.90,
            resolution=(12, 12),
        )

        reflected_beam = self.make_beam(
            start=hit_point,
            end=[8.5, -2.2, 1.35],
            glow_color=red,
            core_color=red_light,
            outer_radius=0.24,
            inner_radius=0.07,
            outer_opacity=0.035,
            inner_opacity=0.55,
            resolution=(12, 12),
        )

        # ====================================================
        # SMALL INTERACTION PATCH ON SURFACE
        # ====================================================

        interaction = Circle(radius=0.42)
        interaction.set_fill(red_light, opacity=0.10)
        interaction.set_stroke(width=0, opacity=0)
        interaction.move_to([hit_point[0], hit_point[1], 0.001])

        # ====================================================
        # STATIC IMAGE
        # ====================================================

        self.add(
            substrate,
            pillars,
            interaction,
            incident_beam,
            reflected_beam,
        )