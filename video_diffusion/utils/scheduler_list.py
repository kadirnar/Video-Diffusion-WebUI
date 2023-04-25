from diffusers import (
    DDIMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
)

diff_scheduler_list = [
    "DDIM",
    "EulerA",
    "Euler",
    "LMS",
    "Heun",
    "UniPC",
]


def get_scheduler_list(pipe, scheduler):
    if scheduler == diff_scheduler_list[0]:
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    elif scheduler == diff_scheduler_list[1]:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    elif scheduler == diff_scheduler_list[2]:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    elif scheduler == diff_scheduler_list[3]:
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

    elif scheduler == diff_scheduler_list[4]:
        pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)

    return pipe
