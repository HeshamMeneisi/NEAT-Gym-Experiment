from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool.gym_forward_walker import RoboschoolForwardWalker
import numpy as np


class ForwardServoWalker(RoboschoolForwardWalker):

    def apply_action(self, a):
        assert(np.isfinite(a).all())
        for n,j in enumerate(self.ordered_joints):
            an = float(np.clip(a[n], -1, +1))
            # j.set_motor_torque( self.power*j.power_coef*an )
            cp = j.current_position()[0]

            label = j.name.split('_')[0]
            low, high, pwm = self.servo_settings[label]

            p = (an + 1) / 2.0 * (high - low) + low

            j.set_servo_target(p, 1, 10, self.power*pwm)

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost


        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        self.rewards = [
            alive,
            progress,
            joints_at_limit_cost,
            feet_collision_cost
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), bool(done), {}


class ForwardServoWalkerMujocoXML(ForwardServoWalker, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        ForwardServoWalker.__init__(self, power)


class MiKoBot(ForwardServoWalkerMujocoXML):
    # Up-to-date model: https://pastebin.com/raw/EMqf2nAR
    foot_list = ['front_left_foot', 'front_right_foot', 'back_right_foot', 'back_left_foot']

    servo_settings = {'hip': (-np.pi/2, np.pi/2, 1), 'ankle': (0, 0.35, 2)}

    def __init__(self):
        ForwardServoWalkerMujocoXML.__init__(self, "miko.xml", "torso", action_dim=8, obs_dim=28, power=80)

    def alive_bonus(self, z, rpy):
        thresh = np.pi / 3
        if abs(rpy[0]) > thresh or abs(rpy[1]) > thresh:
            print("dead")
            return -1
        return 1
