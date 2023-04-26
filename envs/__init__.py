from gymnasium.envs.registration import register

register(
    id="motndp_amsterdam-v0",
    entry_point="motndp.motndp:MOTNDP",
)
