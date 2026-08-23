"""
Metasurface + incident beam — dark-glass scientific style.

Render (still frame, Cairo renderer only — see below):
    manim -s -qh blog_intro_3d.py BlogIntro3D
    manim -s -r 3840,2160 blog_intro_3d.py BlogIntro3D   # 4K blog asset

DO NOT pass --renderer=opengl. This scene is written against the Cairo
renderer on purpose. Cairo has no depth buffer and no lighting model, so
everything below fakes both by hand: it shades each individual quad of each
Surface, culls back faces itself, and overrides Cairo's painter-sort keys via
`z_index_group`. Under the OpenGL renderer a Surface is a GPU mesh with real
depth testing and real shading, so none of those hooks exist (the first thing
that trips is `OpenGLCamera.set_zoom`) and none of them would be wanted. The
scene raises early with this explanation if the OpenGL renderer is active.

Key ideas vs. the previous version
----------------------------------
1. LIGHTING: Manim's Cairo 3D renderer has no real lighting model, so we do our
   own. Every face of every Surface/Prism gets a Lambertian shade computed from
   its geometric normal and a virtual light direction. Manim's own darkening
   pass is switched off (shading_factor = 0) so it cannot double-darken us.
2. NO WIREFRAME: Surfaces in Cairo mode are drawn as a mesh of quads. Any stroke
   at all reads as a wireframe, and stroke width 0 leaves background-colored
   hairline seams. Fix: stroke of the *same* shaded color, width ~1 -> seams are
   filled, mesh disappears. Back faces are culled so the silhouette is clean.
3. BEAM: instead of one opaque pipe, a stack of concentric shells with an
   opacity falloff (soft edge), no end caps, plus a physically reflected ray and
   a faint transmitted ray, and a soft glow pool at the interaction point.
4. SUBSTRATE: a finite, opaque, shaded slab whose top face is exactly z = 0, so
   pillars sit on it instead of being sliced by a giant translucent plane.
"""

from pathlib import Path

import numpy as np
import yaml
from manim import *

# ============================================================
# LOAD STYLE
# ============================================================

STYLE_PATH = Path(__file__).with_name("visual-style_dark.yml")

with open(STYLE_PATH, "r", encoding="utf-8") as f:
    STYLE = yaml.safe_load(f)

config.background_color = STYLE["canvas"]["background"]

PAL = STYLE["colors"]["palette"]
NEU = STYLE["colors"]["neutral"]


# ============================================================
# CAMERA / LIGHT SETUP
# ============================================================

PHI = 64 * DEGREES

# Azimuth matters for more than taste: near -45 deg the lattice diagonal runs
# almost parallel to the view direction, so neighbouring pillars sit at nearly
# the same camera depth and the painter sort starts swapping them (that is what
# produced the stepped, bitten-off pillar edges). -60 deg gives every lattice
# neighbour a clear depth separation.
THETA = -60 * DEGREES

# Slight zoom-out so the composition breathes (style: generous whitespace).
# (>1 zooms in, <1 zooms out. The old `camera.scale(0.60)` was the reason the
# previous frame was cropped so hard.)
CAMERA_ZOOM = 0.90

# Virtual light: high, and swung away from the camera azimuth so each pillar
# gets a lit side, a shaded side and a bright top. A light sitting near the
# camera axis flattens everything into one pale tone.
LIGHT_PHI = 40 * DEGREES
LIGHT_THETA = THETA - 40 * DEGREES

# Highlight colour for the glassy specular term (style: cream neutral).
HIGHLIGHT = "#F3EEE4"


def spherical_dir(phi: float, theta: float) -> np.ndarray:
    return np.array(
        [
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        ]
    )


VIEW_DIR = spherical_dir(PHI, THETA)      # scene -> camera
LIGHT_DIR = spherical_dir(LIGHT_PHI, LIGHT_THETA)
HALF_DIR = (LIGHT_DIR + VIEW_DIR) / np.linalg.norm(LIGHT_DIR + VIEW_DIR)


# ============================================================
# SHADING HELPERS
# ============================================================


def newell_normal(points: np.ndarray) -> np.ndarray:
    """Robust polygon normal (works for bezier-sampled quads too)."""
    p = np.asarray(points, dtype=float)
    if len(p) < 3:
        return np.array([0.0, 0.0, 1.0])

    q = np.roll(p, -1, axis=0)
    n = np.array(
        [
            np.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
            np.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
            np.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
        ]
    )
    mag = np.linalg.norm(n)
    if mag < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return n / mag


def shade_solid(
    mob,
    shadow_color,
    lit_color,
    opacity=1.0,
    ambient=0.30,
    top_boost=1.0,
    cull_backfaces=True,
    seam_width=1.1,
    tint=None,
    tint_amount=0.0,
    origin=None,
    gamma=1.5,
    specular=0.0,
    shininess=30.0,
):
    """
    Per-face Lambertian shading for a convex Surface / Prism.

    - `ambient` keeps unlit faces readable instead of black.
    - `seam_width`: stroke in the face's own color hides the quad mesh.
    - back faces are hidden, which gives a clean silhouette with no wireframe.
    - `tint` blends a colour into up-facing faces only (used to give the
      substrate a glass-blue top without touching its neutral side walls).
    """
    center = mob.get_center() if origin is None else np.asarray(origin, dtype=float)

    for face in mob.family_members_with_points():
        pts = face.points
        n = newell_normal(pts)

        # orient normal outward
        if np.dot(n, face.get_center() - center) < 0:
            n = -n

        if cull_backfaces and np.dot(n, VIEW_DIR) < -0.02:
            face.set_fill(opacity=0.0)
            face.set_stroke(width=0.0, opacity=0.0)
            continue

        diffuse = max(0.0, float(np.dot(n, LIGHT_DIR))) ** gamma
        intensity = ambient + (1.0 - ambient) * diffuse

        # brighten near-horizontal (up-facing) faces: reads as top illumination
        upness = max(0.0, float(n[2]))
        intensity *= 1.0 + (top_boost - 1.0) * upness
        intensity = float(np.clip(intensity, 0.0, 1.0))

        col = interpolate_color(
            ManimColor(shadow_color), ManimColor(lit_color), intensity
        )
        if tint is not None and tint_amount > 0.0:
            col = interpolate_color(col, ManimColor(tint), tint_amount * upness)

        if specular > 0.0:
            spec = specular * max(0.0, float(np.dot(n, HALF_DIR))) ** shininess
            col = interpolate_color(
                col, ManimColor(HIGHLIGHT), float(np.clip(spec, 0.0, 1.0))
            )

        face.set_fill(col, opacity=opacity)
        face.set_stroke(col, width=seam_width, opacity=opacity)

    return mob


def slab_faces(width, depth_, thickness, z_top=0.0, border=0.07,
               res_top=(52, 36), res_side=(40, 3)):
    """
    A slab built from *subdivided* Surfaces instead of a single Prism.

    Why: the Cairo renderer painter-sorts each drawn piece by its centre, so one
    huge quad for the slab's top face gets sorted at the slab's centre and paints
    straight over every pillar standing farther back than that centre. Cutting
    the top and the side walls into many small quads makes each piece sort where
    it actually is, and the occlusion becomes correct.

    Returns (body_parts, edge_parts). The luminous rim is a thin strip of real
    geometry rather than a stroked Rectangle: a separate outline mobject sits a
    hair above the top plane, so the plate's own quads get sorted in front of it
    and chop it into dashes.
    """
    w, d = width / 2.0, depth_ / 2.0
    z_bot = z_top - thickness
    iw, idp = w - border, d - border

    def plane(u_range, v_range, res, mapper):
        return Surface(
            mapper,
            u_range=u_range,
            v_range=v_range,
            resolution=res,
            checkerboard_colors=False,
        )

    body = [
        plane([-iw, iw], [-idp, idp], res_top,
              lambda u, v: np.array([u, v, z_top]))
    ]
    edges = []

    for s in (-1.0, 1.0):
        # top-face border strips
        edges.append(
            plane([-w, w], [s * d - s * border, s * d], (res_top[0], 1),
                  lambda u, v: np.array([u, v, z_top]))
        )
        edges.append(
            plane([s * w - s * border, s * w], [-idp, idp], (1, res_top[1]),
                  lambda u, v: np.array([u, v, z_top]))
        )
        # side walls
        body.append(
            plane([-w, w], [z_bot, z_top], res_side,
                  lambda u, v, s=s: np.array([u, s * d, v]))
        )
        body.append(
            plane([-d, d], [z_bot, z_top], res_side,
                  lambda u, v, s=s: np.array([s * w, u, v]))
        )

    return body, edges


def sort_as_one(mob, point, bias=0.0):
    """
    Force every face of `mob` to be depth-sorted as a single unit, at `point`
    pushed `bias` units toward the camera.

    The Cairo renderer takes each drawn piece's sort key from
    `get_z_index_reference_point()`, which honours a `z_index_group` override.
    Two things come out of this:

    * A convex mobject with its back faces culled has no self-overlap, so one
      shared key is perfectly fine for it.
    * The small camera-ward bias makes a pillar unambiguously win against the
      flat top plate it stands on. Without it, the plate quad that straddles a
      pillar's base can sort in front of the pillar and bite a notch out of it.
    """
    anchor = VectorizedPoint(np.asarray(point, dtype=float) + VIEW_DIR * bias)
    for face in mob.family_members_with_points():
        face.z_index_group = anchor
    return anchor


def flat_layer(mob, color, opacity, stroke_color=None, stroke_width=0.0):
    """Depth-sorted flat element (rims, glow pools, haze)."""
    mob.set_fill(color, opacity=opacity)
    if stroke_color is None:
        mob.set_stroke(width=0.0, opacity=0.0)
    else:
        mob.set_stroke(stroke_color, width=stroke_width, opacity=1.0)
    mob.set_shade_in_3d(True)
    return mob


# ============================================================
# SCENE
# ============================================================


class BlogIntro3D(ThreeDScene):

    # --------------------------------------------------------
    # BEAM
    # --------------------------------------------------------

    def make_beam(
        self,
        start,
        end,
        core_color,
        glow_color,
        core_radius=0.035,
        core_opacity=0.95,
        shells=(0.075, 0.13, 0.21, 0.33),
        shell_opacities=(0.26, 0.13, 0.065, 0.028),
        taper=2.4,
        resolution=(256, 256),
    ):
        """
        A soft, organic beam: bright thin core + concentric glow shells with an
        opacity falloff, no end caps, no contrasting stroke.

        `taper` pulls each shell back from both ends by taper * its radius, so
        the wide faint glow stops short of the thin bright core. The tip fades
        to a point instead of being chopped off by a flat disc — that hard cut
        was what made the old single-cylinder beam look synthetic.
        """
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)

        direction = end - start
        length = float(np.linalg.norm(direction))
        direction = direction / length

        beam = VGroup()

        # outer shells first (painter-friendly), bright core last
        layers = list(zip(shells, shell_opacities, [glow_color] * len(shells)))
        layers.sort(key=lambda t: -t[0])
        layers.append((core_radius, core_opacity, core_color))

        for radius, opacity, color in layers:
            pull = taper * radius
            pull = min(pull, 0.35 * length)
            seg_len = length - 2.0 * pull
            seg_mid = start + direction * (pull + seg_len / 2.0)

            shell = Cylinder(
                radius=radius,
                height=seg_len,
                direction=direction,
                resolution=resolution,
                show_ends=False,
                checkerboard_colors=False,
            )
            shell.move_to(seg_mid)

            for face in shell.family_members_with_points():
                face.set_fill(color, opacity=opacity)
                face.set_stroke(color, width=0.8, opacity=opacity)

            beam.add(shell)

        return beam

    def glow_pool(self, point, color, radii=(0.30, 0.55, 0.95, 1.5),
                  opacities=(0.30, 0.15, 0.07, 0.03), z=0.004):
        """Soft falloff patch on the interaction plane."""
        pool = VGroup()
        for r, o in zip(reversed(radii), reversed(opacities)):
            disc = Circle(radius=r)
            flat_layer(disc, color, o)
            disc.move_to([point[0], point[1], z])
            pool.add(disc)
        return pool

    # --------------------------------------------------------
    # CONSTRUCT
    # --------------------------------------------------------

    def construct(self):

        # ====================================================
        # CAMERA — and disable Manim's own darkening pass so our
        # per-face shading is what actually reaches the canvas.
        #
        # ThreeDCamera.modified_rgbas() darkens every face whose normal points
        # away from its (fixed, ad-hoc) light source. That pass is the main
        # reason the previous frame read as near-black. `shade_in_3d` stays
        # True because the Cairo renderer uses it for depth sorting.
        # ====================================================

        if config.renderer != RendererType.CAIRO:
            raise RuntimeError(
                "BlogIntro3D targets the Cairo renderer. It fakes lighting and "
                "depth sorting by hand (per-face shading, back-face culling, "
                "z_index_group sort keys), and those hooks do not exist in the "
                "OpenGL pipeline. Drop --renderer=opengl and render with e.g. "
                "`manim -s -r 3840,2160 blog_intro_3d.py BlogIntro3D`."
            )

        self.set_camera_orientation(phi=PHI, theta=THETA)

        cam = self.renderer.camera
        if hasattr(cam, "set_zoom"):
            cam.set_zoom(CAMERA_ZOOM)
        cam.should_apply_shading = False

        # ====================================================
        # COLORS
        # ====================================================

        blue = PAL["blue"]
        red = PAL["red"]

        pillar_lit = blue["bright"]       # #7AD7EE
        pillar_mid = blue["bright"]       # #7AD7EE
        pillar_dark = blue["glass"]       # #183A46

        rim_color = blue["bright"]

        slab_lit = NEU[400]               # #8A8985
        slab_dark = "#0E1318"

        beam_core = "#FFE3DE"
        beam_glow = red["main"]           # #FC6255

        # ====================================================
        # LAYOUT
        # ====================================================

        spacing = 1.62
        n_cols, n_rows = 7, 5

        xs = (np.arange(n_cols) - (n_cols - 1) / 2) * spacing
        ys = (np.arange(n_rows) - (n_rows - 1) / 2) * spacing

        slab_w = (n_cols - 1) * spacing + 2.0
        slab_d = (n_rows - 1) * spacing + 2.0
        slab_t = 0.55

        base_radius = 0.36

        # ====================================================
        # SUBSTRATE — finite, opaque, top face exactly at z = 0
        # ====================================================

        slab_body, slab_edges = slab_faces(slab_w, slab_d, slab_t, z_top=0.0)
        slab_origin = np.array([0.0, 0.0, -slab_t / 2])

        substrate = VGroup(*slab_body)
        for part in substrate:
            shade_solid(
                part,
                shadow_color=slab_dark,
                lit_color=slab_lit,
                opacity=1.0,
                ambient=0.22,
                top_boost=1.0,
                seam_width=1.2,
                tint=blue["glass"],   # glass-blue top face, neutral side walls
                tint_amount=0.75,
                origin=slab_origin,
            )

        # luminous rim (style: darker fill, brighter edge)
        slab_rim = VGroup(*slab_edges)
        for part in slab_rim:
            shade_solid(
                part,
                shadow_color=blue["main"],
                lit_color=blue["bright"],
                opacity=1.0,
                ambient=0.80,
                seam_width=1.4,
                origin=slab_origin,
            )

        # NOTE: no separate translucent overlay plane here on purpose. A single
        # large flat quad gets depth-sorted by its centre, so it slices through
        # any pillar whose centre is farther than the quad's — the top face of
        # the slab is tinted instead.

        # ====================================================
        # PILLARS — smooth, shaded, no wireframe
        # ====================================================

        pillars = VGroup()
        rims = VGroup()

        def pillar_geom(i, j):
            radius = base_radius + 0.035 * np.sin(0.8 * i + 0.5 * j)
            height = 1.10 + 0.12 * np.cos(0.7 * i - 0.6 * j)
            return float(radius), float(height)

        # Center the beam: aim at the grid centre pillar.
        hit_i, hit_j = n_cols // 2, n_rows // 2
        hit_r, hit_h = pillar_geom(hit_i, hit_j)
        hit = np.array([float(xs[hit_i]), float(ys[hit_j]), hit_h])

        # LOCAL RESONANCE: the incident field couples into the pillars around
        # the hit point and rings there, decaying laterally. Each pillar gets a
        # resonance amplitude from its distance to the hit — that amplitude
        # drives a warm body tint, a warm rim, and a glow cap, so the energy
        # visibly lives IN the resonators near the surface, not in free space.
        resonance_length = 1.4 * spacing

        def resonance_amp(x, y):
            dist = np.hypot(x - hit[0], y - hit[1])
            return float(np.exp(-dist / resonance_length))

        entries = []

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                radius, height = pillar_geom(i, j)
                entries.append(
                    (float(x), float(y), radius, height, resonance_amp(x, y))
                )

        # painter's order: farthest from camera first
        entries.sort(key=lambda e: np.dot(np.array([e[0], e[1], 0.0]), VIEW_DIR))

        # tiny sink into the slab hides the seam where a pillar meets the plate
        sink = 0.02

        resonance_caps = VGroup()

        for x, y, radius, height, amp in entries:
            pillar = Cylinder(
                radius=radius,
                height=height + sink,
                direction=OUT,
                resolution=(256, 256),   # smooth silhouette
                show_ends=True,
                checkerboard_colors=False,
            )
            pillar.move_to([x, y, (height - sink) / 2])
            shade_solid(
                pillar,
                shadow_color=pillar_dark,
                lit_color=pillar_lit,
                opacity=0.94,
                ambient=0.20,
                top_boost=1.25,
                seam_width=1.1,
                gamma=1.8,
                specular=0.85,
                shininess=30.0,
                # resonating pillars glow warm from within, fading with distance
                tint=red["light"] if amp > 0.05 else None,
                tint_amount=0.60 * amp,
            )
            anchor = sort_as_one(pillar, [x, y, 0.0], bias=0.55)
            pillars.add(pillar)

            rim = Circle(radius=radius)
            rim_col = interpolate_color(
                ManimColor(blue["bright"]), ManimColor(red["light"]), amp
            )
            flat_layer(rim, pillar_mid, 0.0,
                       stroke_color=rim_col,
                       stroke_width=1.4 + 1.0 * amp)
            rim.set_stroke(opacity=0.75 + 0.2 * amp)
            rim.move_to([x, y, height + 0.004])
            # share the pillar's sort key, and sit after it in scene order, so
            # the lit rim always lands on its own pillar's top edge
            for face in rim.family_members_with_points():
                face.z_index_group = anchor
            rims.add(rim)

            # glow cap: soft red pool sitting on each resonating pillar's top,
            # amplitude-scaled — the standing field stored in the resonator
            if amp > 0.10:
                cap = self.glow_pool(
                    np.array([x, y, height]),
                    red["light"],
                    radii=(radius * 0.45, radius * 0.75, radius * 1.05),
                    opacities=(0.40 * amp, 0.22 * amp, 0.10 * amp),
                    z=height + 0.008,
                )
                for face in cap.family_members_with_points():
                    face.z_index_group = anchor
                resonance_caps.add(cap)

        # ====================================================
        # BEAM GEOMETRY — hit the metasurface plane, reflect properly
        # ====================================================

        # `hit` was set with the pillar grid above: the beam lands exactly on one
        # pillar's top face, so the interaction sits on the metasurface instead
        # of floating above it.
        incident_dir = np.array([0.92, -0.34, -0.42])
        incident_dir /= np.linalg.norm(incident_dir)

        entry = hit - incident_dir * 11.0

        normal = np.array([0.0, 0.0, 1.0])
        reflect_dir = incident_dir - 2.0 * np.dot(incident_dir, normal) * normal
        reflect_end = hit + reflect_dir * 9.0

        incident = self.make_beam(
            entry,
            hit,
            core_color=beam_core,
            glow_color=beam_glow,
            core_radius=0.070,     # thicker
            core_opacity=0.95,
            shells=(0.13, 0.22, 0.35, 0.55),  # thicker shells
            shell_opacities=(0.24, 0.12, 0.06, 0.025),
        )

        reflected = self.make_beam(
            hit,
            reflect_end,
            core_color=beam_core,
            glow_color=beam_glow,
            core_radius=0.050,     # thicker
            core_opacity=0.65,
            shells=(0.10, 0.17, 0.28, 0.44),  # thicker shells
            shell_opacities=(0.15, 0.075, 0.038, 0.016),
        )

        # ====================================================
        # INTERACTION GLOW — kept tight to the illuminated pillar top
        # ====================================================

        pool = self.glow_pool(
            hit,
            red["light"],
            radii=(hit_r * 0.55, hit_r * 0.95, hit_r * 1.5, hit_r * 2.3),
            opacities=(0.36, 0.18, 0.075, 0.028),
            z=hit_h + 0.012,
        )

        hot_spot = Dot3D(point=hit, radius=0.075, color=beam_core,
                         resolution=(256, 256), checkerboard_colors=False)
        for face in hot_spot.family_members_with_points():
            face.set_fill(beam_core, opacity=1.0)
            face.set_stroke(beam_core, width=1.0, opacity=1.0)

        # ====================================================
        # NEAR-FIELD HOT SPOTS — the field trapped in the gaps
        # ====================================================
        # In a resonant metasurface the enhanced field is concentrated in the
        # narrow gaps between neighbouring resonators (near-field coupling).
        # Faint vertical glow columns in the four gaps around the hit pillar,
        # confined below the pillar tops — nothing radiates away.

        gap_fields = VGroup()

        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = hit_i + di, hit_j + dj
            if not (0 <= ni < n_cols and 0 <= nj < n_rows):
                continue

            nx, ny = float(xs[ni]), float(ys[nj])
            n_r, n_h = pillar_geom(ni, nj)
            amp = resonance_amp(nx, ny)

            gap_mid = np.array([(hit[0] + nx) / 2.0, (hit[1] + ny) / 2.0, 0.0])
            gap_h = 0.85 * min(hit_h, n_h)

            for gr, go in ((0.09, 0.26), (0.16, 0.12), (0.25, 0.05)):
                column = Cylinder(
                    radius=gr,
                    height=gap_h,
                    direction=OUT,
                    resolution=(256, 256),
                    show_ends=False,
                    checkerboard_colors=False,
                )
                column.move_to([gap_mid[0], gap_mid[1], gap_h / 2.0])
                for face in column.family_members_with_points():
                    face.set_fill(red["light"], opacity=go * amp)
                    face.set_stroke(red["light"], width=1.2, opacity=go * amp)
                gap_fields.add(column)

        # ====================================================
        # STATIC IMAGE
        # ====================================================

        self.add(
            substrate,
            slab_rim,
            pillars,
            rims,
            resonance_caps,
            gap_fields,
            pool,
            incident,
            reflected,
            hot_spot,
        )
