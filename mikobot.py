from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool.gym_forward_walker import RoboschoolForwardWalker
import numpy as np
import math


class ForwardServoWalker(RoboschoolForwardWalker):

    def __init__(self, power, state_dim):
        self.last_state = np.random.rand(state_dim)
        self.last_dist = 0
        self.still_counter = 0
        RoboschoolForwardWalker.__init__(self, power)

    def apply_action(self, a):
        assert(np.isfinite(a).all())

        if self.frame < 20:
            return
        elif self.frame == 20:
            self.no_death = 'no_death' in self._spec.tags

        for n, j in enumerate(self.ordered_joints):
            an = float(np.clip(a[n], -1, +1))
            cp = j.current_position()[0]

            label = j.name.split('_')[0]
            low, high, pwm, kp, kd = self.servo_settings[label]

            p = (an + 1) / 2.0 * (high - low) + low

            j.set_servo_target(p, kp, kd, self.power*pwm)

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0 and not self.no_death
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        jump_cost = 0.0
        air_feet = 0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0

        air_feet = len(self.feet_contact) - np.sum(self.feet_contact)

        if air_feet > 3 and self.frame > 20:
            jump_cost = -1

        electricity_cost = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        self.rewards = [
            alive,
            progress,
            joints_at_limit_cost,
            jump_cost
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done += done   # 2 == 1+True
        self.reward = np.sum(self.rewards)
        if self.reward > 2000:
            print("")
        self.HUD(state, a, done)
        return state, sum(self.rewards), bool(done), {}

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

        if abs(np.sum(state - self.last_state)) < 0.01 or self.last_dist - self.walk_target_dist < 0.01:
            self.still_counter += 1
        else:
            self.still_counter = 0
        self.last_state = state
        self.last_dist = self.walk_target_dist

        return state


class ForwardServoWalkerMujocoXML(ForwardServoWalker, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        ForwardServoWalker.__init__(self, power, obs_dim)


class MiKoBot(ForwardServoWalkerMujocoXML):
    # Up-to-date model: https://pastebin.com/raw/EMqf2nAR
    foot_list = ['front_left_foot', 'front_right_foot', 'back_right_foot', 'back_left_foot']

    servo_settings = {'hip': (-np.pi/2, np.pi/2, 1, 0.08, 1), 'ankle': (0.35, 0, 1.5, 0.15, 1.3)}

    def __init__(self):
        ForwardServoWalkerMujocoXML.__init__(self, "miko.xml", "torso", action_dim=8, obs_dim=10, power=200)

    def alive_bonus(self, z, _):
        if abs(self.body_xyz[1]) > 2:
            return -1

        thresh = np.pi / 3
        rpy = self.body_rpy
        if abs(rpy[0]) > thresh or abs(rpy[1]) > thresh or self.still_counter > 100:
            self.still_counter = 0
            return -1
        return 1
