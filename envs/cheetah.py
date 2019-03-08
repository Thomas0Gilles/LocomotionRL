import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import scipy

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.res=20
        self.n_points=200
        self.has_terrain = False
        mujoco_env.MujocoEnv.__init__(self, '/Users/tgill/OneDrive/Documents/GD_AI/LocomotionRL-master/envs/assets/half_cheetah_hfield.xml', 5)
        utils.EzPickle.__init__(self)
#        self.terrain = self.height2terrain()
#        self.has_terrain = True

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
    
    def _get_hmap(self):
        x = self.sim.data.qpos[0]
        x = int(x)
        height_map = self.terrain[x:x+200][0]
        return height_map

    def _get_obs(self):
        if self.has_terrain:
            hmap = self._get_hmap()
        else:
            hmap = np.zeros(self.n_points)
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
           # hmap,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        
    def height2terrain(self):
        hfield = self.sim.model.hfield_data
        print(hfield)
        hfield = hfield.reshape(self.sim.model.hfield_nrow[0], self.sim.model.hfield_ncol[0])
        x_size, y_size = self.sim.model.hfield_size[0][:2]
        terrain = scipy.misc.imresize(hfield, (int(x_size)*self.res, int(y_size)*self.res))
        return terrain