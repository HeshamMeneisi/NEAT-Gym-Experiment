from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool.gym_forward_walker import RoboschoolForwardWalker
import numpy as np


class ForwardServoWalker(RoboschoolForwardWalker):

    def __init__(self, power, state_dim):
        self.last_state = np.random.rand(state_dim)
        self.still_counter = 0
        RoboschoolForwardWalker.__init__(self, power)

    def apply_action(self, a):
        assert(np.isfinite(a).all())

        for n, j in enumerate(self.ordered_joints):
            an = float(np.clip(a[n], -1, +1))
            cp = j.current_position()[0]

            label = j.name.split('_')[0]
            low, high, pwm, kp, kd = self.servo_settings[label]

            p = (an + 1) / 2.0 * (high - low) + low

            j.set_servo_target(p, kp, kd, self.power*pwm)

    def calc_state(self):
        j = np.array([j.current_relative_position()[0] for j in self.ordered_joints], dtype=np.float32)
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = np.array([j.current_relative_position()[1] for j in self.ordered_joints], dtype=np.float32)
        self.joints_at_limit = np.count_nonzero(np.abs(j) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
        parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        r, p, yaw = self.body_rpy
        if self.initial_z == None:
            self.initial_z = z
        self.walk_target_theta = np.arctan2(self.walk_target_y - self.body_xyz[1],
                                            self.walk_target_x - self.body_xyz[0])
        self.walk_target_dist = np.linalg.norm(
            [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]])
        self.angle_to_target = self.walk_target_theta - yaw

        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [0, 0, 1]]
        )

        state = np.concatenate([[self.walk_target_x, self.walk_target_y], np.clip(j, -5, +5)])

        if abs(np.sum(state - self.last_state)) < 0.01:
            self.still_counter += 1
        else:
            self.still_counter = 0
        self.last_state = state

        return state


class ForwardServoWalkerMujocoXML(ForwardServoWalker, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        ForwardServoWalker.__init__(self, power, obs_dim)


class MiKoBot(ForwardServoWalkerMujocoXML):
    # Up-to-date model: https://pastebin.com/raw/EMqf2nAR
    foot_list = ['front_left_foot', 'front_right_foot', 'back_right_foot', 'back_left_foot']

    servo_settings = {'hip': (-np.pi/2, np.pi/2, 1, 0.1, 1), 'ankle': (0, 0.35, 2, 0.1, 0.5)}

    def __init__(self):
        ForwardServoWalkerMujocoXML.__init__(self, "miko.xml", "torso", action_dim=8, obs_dim=10, power=180)

    def alive_bonus(self, z, _):
        thresh = np.pi / 3
        rpy = self.body_rpy
        if abs(rpy[0]) > thresh or abs(rpy[1]) > thresh or self.still_counter > 100:
            self.still_counter = 0
            return -1
        return 1
