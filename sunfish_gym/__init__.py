from gym.envs.registration import register

register(
	id='Chess-v0',
	entry_point='sunfish_gym.sunfish_env:SunfishEnv',
	kwargs={
		'opponent' : None,
		'play_as_white' : None
	}
)