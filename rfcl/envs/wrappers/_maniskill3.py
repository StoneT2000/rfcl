from rfcl.envs.wrappers.curriculum import InitialStateWrapper


class ManiSkill3InitialStateWrapper(InitialStateWrapper):
    def set_env_state(self, state):
        if len(state.shape) == 1:
            state = state.reshape(1, -1)
        return self.env.unwrapped.set_state(state)

    def get_env_obs(self):
        return self.env.unwrapped.get_obs()[0]
