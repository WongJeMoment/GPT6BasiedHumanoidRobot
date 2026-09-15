"""终端状态快照与手动 TXT 导出；记录仅驻留内存，按 s 才写文件。"""
from datetime import datetime
from pathlib import Path
import os
import select
import sys
import termios
import tty


def capture_terminal_state(env):
    """在自动复位前读取物理状态，返回与仿真张量完全脱离的快照。"""
    import torch

    objects = torch.stack([obj.data.root_state_w for obj in env.objects], dim=1)
    box = objects[env.indices, env.active_object.clamp_min(0)].clone()
    robot = env.robot.data.root_state_w.clone()
    box[:, :3] -= env.scene.env_origins
    robot[:, :3] -= env.scene.env_origins
    fields = dict(
        active=env.active_object, throw=env.throw_count,
        episode_time=env.episode_length_buf * env.step_dt,
        box=box, robot=robot, gravity_z=env.robot.data.projected_gravity_b[:, 2],
        fallen=env.fallen, terminated=env.reset_terminated, timeout=env.reset_time_outs,
    )
    task = env.shelf_task
    if task is not None:
        from .shelf_task import box_geometry

        extent, _ = box_geometry(box[:, 3:7], task.half_size)
        forces = env.shelf_sensor.data.force_matrix_w[:, 0].norm(dim=-1)
        fields.update({name: getattr(task, name) for name in (
            "phase", "caught", "dropped", "success", "contact", "robot_touch",
            "supported", "on_shelf", "shelf_first", "catch_time", "settle_time",
        )})
        fields.update(
            bottom=box[:, 2] - extent[:, 2],
            shelf_touch=forces[:, 0] > task.cfg.contact_force,
            contact_forces=torch.stack([
                *(forces[:, group].sum(-1) for group in env.shelf_contact_groups), forces[:, 0],
            ], dim=-1),
            body_forces=forces[:, 1:],
        )
    values = {name: value.detach().cpu().tolist() for name, value in fields.items()}
    records = []
    for index in range(env.num_envs):
        row = {name: value[index] for name, value in values.items()}
        row.update(env=index, sim_time=env.common_step_counter * env.step_dt)
        row["object_name"] = env.settings.objects[row["active"]].name if row["active"] >= 0 else "未投放"
        if task is not None:
            row["touching_bodies"] = [
                name for name, force in zip(env.shelf_robot_bodies, row.pop("body_forces"))
                if row["active"] >= 0 and force > task.cfg.contact_force
            ]
            # 备用箱体未投放时，传感器的残留数据不作为当前接触。
            if row["active"] < 0:
                row["shelf_touch"] = False
                row["contact_forces"] = [0.0] * 4
        records.append(row)
    return records


def _yes(value):
    return "是" if value else "否"


def _vector(values):
    return "(" + ", ".join(f"{value:.3f}" for value in values) + ")"


def format_status(row, episode):
    reasons = []
    if row["fallen"]:
        reasons.append("机器人跌倒")
    if row.get("dropped"):
        reasons.append("箱子掉落/越界")
    if row.get("success"):
        reasons.append("放架成功")
    if row["timeout"]:
        reasons.append("超时")
    if row["terminated"] and not reasons:
        reasons.append("终止")
    header = (f"[仿真 {row['sim_time']:.3f}s | 环境 {row['env']} | 回合 {episode} | "
              f"回合时间 {row['episode_time']:.3f}s | 投放 {row['throw']}]")
    if reasons:
        header += " 回合结束（复位前）: " + "、".join(reasons)
    robot = row["robot"]
    posture = "跌倒" if row["fallen"] else ("直立" if row["gravity_z"] < -0.85 else "倾斜")
    lines = [header, f"  机器人: {posture} | 位置(m)={_vector(robot[:3])} | "
             f"姿态(wxyz)={_vector(robot[3:7])} | 线速度(m/s)={_vector(robot[7:10])} | "
             f"角速度(rad/s)={_vector(robot[10:13])} | 跌倒={_yes(row['fallen'])}"]
    if row["active"] < 0:
        lines.append("  箱子/物体: 未投放")
    else:
        box = row["box"]
        lines.append(f"  箱子/物体: {row['object_name']} | 位置(m)={_vector(box[:3])} | "
                     f"姿态(wxyz)={_vector(box[3:7])} | 线速度(m/s)={_vector(box[7:10])} | "
                     f"角速度(rad/s)={_vector(box[10:13])}")
    if "phase" not in row:
        lines.append("  接触情况=未启用检测 | 是否掉落=未启用判定（当前不是接箱放架场景）")
    else:
        phase = ("等待投放", "等待接住", "已接住/放架中", "静置计时", "成功")[row["phase"]]
        bottom = f"{row['bottom']:.3f}m" if row["active"] >= 0 else "无"
        lines.append(f"  箱子状态: {phase} | 曾接住={_yes(row['caught'])} | "
                     f"已在架面={_yes(row['on_shelf'])} | 是否掉落={_yes(row['dropped'])} | "
                     f"箱底离地={bottom} | 成功={_yes(row['success'])} | "
                     f"接住计时={row['catch_time']:.3f}s | 放稳计时={row['settle_time']:.3f}s")
        contacts = [*row["contact"], row["shelf_touch"]]
        detail = " | ".join(f"{name}={_yes(touch)}({force:.3f}N)" for name, touch, force in
                            zip(("左臂/手", "右臂/手", "躯干", "架面"), contacts, row["contact_forces"]))
        lines.append(f"  接触: {detail} | 机器人任意部位={_yes(row['robot_touch'])} | "
                     f"架面承重={_yes(row['supported'])} | 接住前碰架={_yes(row['shelf_first'])} | "
                     f"接触部位={','.join(row['touching_bodies']) or '无'}")
    return "\n".join(lines)


class TerminalStatusRecorder:
    """每 0.5 仿真秒输出状态，接触/阶段变化与终止事件立即输出并留存。"""

    def __init__(self, directory, config, seed, stream=None, interval=0.5):
        self.directory = Path(directory)
        self.stream = sys.stdout if stream is None else stream
        self.interval = interval
        self.started = datetime.now().astimezone()
        self.header = (f"G1 机器人运行状态记录\n开始时间: {self.started.isoformat()}\n"
                       f"场景: {config}\n随机种子: {seed}\n"
                       "位置相对各环境原点，XYZ 轴与世界坐标一致；四元数顺序 wxyz。\n"
                       "掉落沿用任务判据：箱底触地阈值或越界；机器人跌倒单独记录。\n"
                       f"每 {interval:g} 仿真秒采样；接触、阶段变化及回合结束即时记录。\n"
                       "只有在终端按 s 才导出；文件包含本次运行截至按键时已输出的全部状态。\n")
        self.records = []
        self._last = {}
        self._episodes = {}

    def record(self, rows, force=False):
        for row in rows:
            index = row["env"]
            episode = self._episodes.setdefault(index, 1)
            signature = repr([row.get(key) for key in (
                "active", "throw", "phase", "caught", "dropped", "success", "contact",
                "robot_touch", "shelf_touch", "supported", "on_shelf", "shelf_first",
                "touching_bodies", "fallen", "terminated", "timeout",
            )]) + str(row["gravity_z"] < -0.85)
            previous = self._last.get(index)
            ended = row["terminated"] or row["timeout"]
            if (force or ended or previous is None or signature != previous[1]
                    or row["sim_time"] - previous[0] >= self.interval - 1e-9):
                text = format_status(row, episode)
                self.records.append(text)
                print(text, file=self.stream, flush=True)
                self._last[index] = (row["sim_time"], signature)
            if ended:
                self._episodes[index] += 1

    def handle_keys(self, keys):
        """每次按 s 保存一份快照；其他输入、退出及析构均不会触发写盘。"""
        for key in keys:
            if key.lower() != "s":
                continue
            if not self.records:
                print("尚无状态记录，未创建文件。", file=self.stream, flush=True)
                continue
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"robot_status_{datetime.now():%Y%m%d_%H%M%S_%f}.txt"
                # 排他创建，重复保存也不覆盖已有文件。
                with path.open("x", encoding="utf-8") as output:
                    output.write(self.header)
                    output.write(f"保存时间: {datetime.now().astimezone().isoformat()}\n\n")
                    for record in self.records:
                        output.write(record + "\n\n")
                print(f"已保存 TXT（{len(self.records)} 条状态）: {path.resolve()}", file=self.stream, flush=True)
            except OSError as error:
                print(f"保存失败: {error}；内存记录仍保留，可再次按 s 重试。", file=self.stream, flush=True)


class TerminalKeys:
    """非阻塞读取真实终端；保留 Ctrl+C，退出时恢复原来的终端设置。"""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.fd = None
        self._original = None

    def __enter__(self):
        try:
            fd = self.stream.fileno()
            if not os.isatty(fd):
                return self
            original = termios.tcgetattr(fd)
            self.fd, self._original = fd, original
            # TCSANOW 保留输入队列；cbreak 不需要回车且保留信号键。
            tty.setcbreak(fd, termios.TCSANOW)
        except (AttributeError, ValueError, OSError, termios.error):
            self.close()
        return self

    @property
    def enabled(self):
        return self.fd is not None

    def poll(self):
        if self.fd is None:
            return ""
        try:
            if select.select([self.fd], [], [], 0)[0]:
                return os.read(self.fd, 1024).decode("utf-8", errors="ignore")
        except (OSError, ValueError):
            self.close()
        return ""

    def close(self):
        if self.fd is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self._original)
            except (OSError, termios.error):
                pass
            finally:
                self.fd = None
                self._original = None

    def __exit__(self, *exc):
        self.close()
