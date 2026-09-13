"""地面的分米标尺：纯 USD 视觉网格，不添加刚体、碰撞或训练观测。"""
import math


DM = 0.1
SEGMENTS = {
    "a": ((0, 1), (.6, 1)), "b": ((.6, 1), (.6, .5)),
    "c": ((.6, .5), (.6, 0)), "d": ((0, 0), (.6, 0)),
    "e": ((0, .5), (0, 0)), "f": ((0, 1), (0, .5)), "g": ((0, .5), (.6, .5)),
}
DIGITS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
          "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg", "-": "g"}
LETTERS = {
    "d": [((.6, 0), (.6, 1)), ((0, 0), (.6, 0)), ((0, 0), (0, .6)), ((0, .6), (.6, .6))],
    "m": [((0, 0), (0, .6)), ((0, .6), (.6, .6)), ((.3, 0), (.3, .6)), ((.6, 0), (.6, .6))],
    "X": [((0, 0), (.6, 1)), ((0, 1), (.6, 0))],
    "Y": [((0, 1), (.3, .5)), ((.6, 1), (.3, .5)), ((.3, .5), (.3, 0))],
    " ": [],
}


def ruler_geometry(settings):
    """合并为一个平面网格，精确 0.1 m 间距；数字由向量笔画生成，无字体/贴图依赖。"""
    vertices, counts, indices, colors = [], [], [], []
    minor, major = (.22, .26, .30), (.38, .44, .50)
    ink, accent = (.85, .90, .94), (1.0, .64, .16)
    x_min, x_max = -1.5, math.ceil(settings.distance_range[1] + .5)
    y_min, y_max = -2.0, 2.0

    def polygon(points, color, z):
        start = len(vertices)
        vertices.extend((x, y, z) for x, y in points)
        counts.append(len(points))
        indices.extend(range(start, len(vertices)))
        colors.append(color)

    def line(start, end, width, color, z=.005):
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = math.hypot(dx, dy)
        if norm == 0:
            return
        nx, ny = -dy * width / (2 * norm), dx * width / (2 * norm)
        polygon([(start[0] - nx, start[1] - ny), (end[0] - nx, end[1] - ny),
                 (end[0] + nx, end[1] + ny), (start[0] + nx, start[1] + ny)], color, z)

    def text(label, x, y, height=.14, color=ink):
        for offset, char in enumerate(label):
            strokes = [SEGMENTS[s] for s in DIGITS[char]] if char in DIGITS else LETTERS[char]
            for a, b in strokes:
                line((x + height * (.85 * offset + a[0]), y + height * a[1]),
                     (x + height * (.85 * offset + b[0]), y + height * b[1]),
                     height * .07, color, z=.008)

    polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], (.08, .10, .12), .002)
    # 着色区域与实际采样一致：独立 XY 矩形，或原有的距离/方位角扇区。
    r0, r1 = settings.distance_range
    a0, a1 = settings.azimuth_range
    if settings.launch_lateral_range is not None:
        y0, y1 = settings.launch_lateral_range
        polygon([(r0, y0), (r1, y0), (r1, y1), (r0, y1)], (.14, .23, .22), .003)
    else:
        for i in range(32):
            a, b = a0 + (a1 - a0) * i / 32, a0 + (a1 - a0) * (i + 1) / 32
            polygon([(r0 * math.cos(a), r0 * math.sin(a)), (r1 * math.cos(a), r1 * math.sin(a)),
                     (r1 * math.cos(b), r1 * math.sin(b)), (r0 * math.cos(b), r0 * math.sin(b))],
                    (.14, .23, .22), .003)
    for axis, low, high in ((0, x_min, x_max), (1, y_min, y_max)):
        for tick in range(round(low / DM), round(high / DM) + 1):
            value = tick * DM
            color = major if tick % 10 == 0 else minor
            width = .007 if tick % 10 == 0 else .0025
            ends = ((value, y_min), (value, y_max)) if axis == 0 else ((x_min, value), (x_max, value))
            line(*ends, width, color, z=.0045)
    # 正交标尺的零点对应机器人初始站位；每 1 dm 一刻度，每 5 dm 标数。
    line((0, y_min), (0, y_max), .012, ink)
    line((x_min, 0), (x_max, 0), .012, ink)
    line((0, -1.5), (x_max, -1.5), .014, ink)
    for tick in range(round(x_max / DM) + 1):
        x = tick * DM
        length = .14 if tick % 5 == 0 else .065
        line((x, -1.5), (x, -1.5 - length), .010 if tick % 5 == 0 else .006, ink)
        if tick % 5 == 0:
            label = str(tick)
            text(label, x - len(label) * .055, -1.87)
    line((-1.0, -1.5), (-1.0, 1.5), .014, ink)
    for tick in range(-15, 16):
        y = tick * DM
        length = .12 if tick % 5 == 0 else .06
        line((-1.0, y), (-1.0 - length, y), .010 if tick % 5 == 0 else .006, ink)
        if tick % 5 == 0:
            text(str(tick), -1.44, y - .06, height=.12)
    text("X dm", x_max - .65, -1.30, height=.17)
    text("Y dm", -1.42, 1.73, height=.17)
    for radius in (r0, r1):
        if settings.launch_lateral_range is not None:
            line((radius, y0), (radius, y1), .018, accent, z=.006)
        else:
            for i in range(32):
                a, b = a0 + (a1 - a0) * i / 32, a0 + (a1 - a0) * (i + 1) / 32
                if b > a:
                    line((radius * math.cos(a), radius * math.sin(a)),
                         (radius * math.cos(b), radius * math.sin(b)), .018, accent, z=.006)
        text(f"{round(radius / DM)} dm", radius - .22, -.96, height=.16, color=accent)
    if settings.launch_lateral_range is not None:
        for y in (y0, y1):
            line((r0, y), (r1, y), .018, accent, z=.006)
            text(f"{round(y / DM)} dm", r1 + .12, y - .06, color=accent)
    return vertices, counts, indices, colors


def spawn_ground_ruler(settings):
    from isaaclab.sim.utils import get_current_stage
    from pxr import Gf, UsdGeom

    vertices, counts, indices, colors = ruler_geometry(settings)
    mesh = UsdGeom.Mesh.Define(get_current_stage(), "/World/envs/env_0/GroundRuler")
    mesh.CreatePointsAttr(vertices)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform).Set([Gf.Vec3f(*color) for color in colors])
    mesh.GetPrim().SetCustomDataByKey("ruler_unit", "dm")
    mesh.GetPrim().SetCustomDataByKey("ruler_step_m", DM)
