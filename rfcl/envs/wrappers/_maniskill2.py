from rfcl.envs.wrappers.curriculum import InitialStateWrapper


class ManiSkill2InitialStateWrapper(InitialStateWrapper):
    def set_env_state(self, state):
        return self.env.unwrapped.set_state(state)

    def get_env_obs(self):
        return self.env.unwrapped.get_obs()