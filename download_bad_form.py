import os
import sys
import subprocess
import imageio_ffmpeg

BAD_FORM_DATASET = {
    # --- FIT3D CORE EXERCISES ---
    "pushup": [
        {"url": "https://youtube.com/shorts/yQEx9OC2C3E?si=ocO5ugK5xJkiN1mt", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/1AugTfDd3Yk?si=TSjqV2NSNcKa-B3t", "start": 0, "end": 2},
        {"url": "https://youtu.be/E7Pd5wrC4_c?si=xRuH-No4bbpoVjI2", "start": 41, "end": 46},
        {"url": "https://youtu.be/E7Pd5wrC4_c?si=xRuH-No4bbpoVjI2", "start": 47, "end": 51},
        {"url": "https://youtu.be/E7Pd5wrC4_c?si=xRuH-No4bbpoVjI2", "start": 54, "end": 59},
        {"url": "https://youtube.com/shorts/Bei1RbZdFgI?si=xQGuQ45c4O0GtMPR", "start": 0, "end": 7},
        {"url": "https://youtube.com/shorts/UIcct-7b6oE?si=48r8LFTP3Mk-VtUk", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/udQBmkvYOmo?si=AXzP6-StlfD74XeA", "start": 0, "end": 2}

    ],
    "burpees": [
        {"url": "https://youtube.com/shorts/S6lclRRTjyc?si=V4R7RayWhqlpHgeX", "start": 1, "end": 7},
        {"url": "https://youtube.com/shorts/kS7xbNw4IBU?si=zDmMGUcJxidGo2v8", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/24AwdIlc7WA?si=rKRTr041mneRewMz", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/zlYA1SENYG4?si=FiX08i_KlEK_PWcz", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/oUOeCtL0lg4?si=a4Vso_66fdPETjQK", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/oUOeCtL0lg4?si=a4Vso_66fdPETjQK", "start": 5, "end": 10},
        {"url": "https://youtube.com/shorts/52e6CxY7siw?si=FLPYEl3_j-L0j9mC", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/aiWQ96gU6Vg?si=7P1LEpEjNqG5ABsE", "start": 15, "end": 20},
        {"url": "https://youtube.com/shorts/pHp_Z9jLsUM?si=aZJuK4F_jY6uRMde", "start": 0, "end": 3},
        {"url": "https://youtu.be/W8x9Zt5A2-0?si=8lYR3_YSqXDTbarf", "start": 33, "end": 64}
        
    ],
    "mule_kick": [
        {"url": "https://youtu.be/Y7947iQH4cc?si=zgv8NdyKj7lu7CCs", "start": 0, "end": 5},
        {"url": "https://youtu.be/H8T3PHYUUSo?si=qkO3wI_5WDZZsvtt", "start": 0, "end": 19},
        {"url": "https://youtube.com/shorts/mvAVPx28mn4?si=19E0Jz6rT7A-0JqM", "start": 3, "end": 21},
        {"url": "https://youtu.be/k9nOum4LwtA?si=RgBlh2Yajh51m6Jr", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/_83ens2hJCA?si=mqPj3NWTbyOsnE1A", "start": 4, "end": 9}
        
    ],
    "dumbbell_reverse_lunge": [
        {"url": "https://youtube.com/shorts/-kCI9EuTpZs?si=TOzbBYm2Hrht1aFX", "start": 1, "end": 23},
        {"url": "https://youtube.com/shorts/7g9jihaQT7M?si=9a9ygjKiK-dooK86", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/cPCLYMxjqwA?si=v61HtsS0BwVTqQl_", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/YmQf4vkjyKw?si=vXQZhOzu2gb2K2R4", "start": 4, "end": 10},
        {"url": "https://youtube.com/shorts/YmQf4vkjyKw?si=vXQZhOzu2gb2K2R4", "start": 16, "end": 18},
        {"url": "https://youtube.com/shorts/YmQf4vkjyKw?si=vXQZhOzu2gb2K2R4", "start": 23, "end": 26}
        
    ],
    "dumbbell_overhead_shoulder_press": [
        {"url": "https://youtube.com/shorts/s6hDp6prx4Y?si=dbmBiPAqJbOri_e3", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/daoVqvBm7I0?si=TfoSwJnl6SWKtyxr", "start": 0, "end": 4}
        
    ],
    "dumbbell_hammer_curls": [
        {"url": "https://youtu.be/rbOhYRsd9Ak?si=hqAKiiQDBMJbw-kA", "start": 109, "end": 122},
        {"url": "https://youtu.be/rbOhYRsd9Ak?si=hqAKiiQDBMJbw-kA", "start": 2, "end": 7}

    ],
    "band_pull_apart": [
        {"url": "https://youtube.com/shorts/SLFDrgKarUs?si=xibBPlejx6h74nrV", "start": 4, "end": 16},
        {"url": "https://youtube.com/shorts/QHy6QzOocbM?si=QMQ8xXywhRnWRAiJ", "start": 0, "end": 3},
        {"url": "https://youtu.be/mt73tb2npJM?si=p6QLUNmhA4eWsZD-", "start": 43, "end": 50},
        {"url": "https://youtube.com/shorts/f3_6vLXKL1I?si=oNbpJWjVsJQK31j3", "start": 19, "end": 26},
        {"url": "https://youtube.com/shorts/Nwm80m3TjbE?si=8Ls42kqYNPpRHFMU", "start": 7, "end": 9},
        {"url": "https://youtube.com/shorts/6eictYvPulw?si=8w5w5htTkS7XixMR", "start": 0, "end": 13},
        {"url": "https://youtube.com/shorts/KG_SamTu_UY?si=4SA9q8xt8R6FOJ9I", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/9PEryeipUu4?si=nUfc5MP2fpgLCCCu", "start": 0, "end": 13},
        {"url": "https://youtube.com/shorts/KMOLpSSqCL8?si=s7fLfoS8A14mD_QS", "start": 0, "end": 4}
        
    ],
    "barbell_row": [
        {"url": "https://youtu.be/qXrTDQG1oUQ?si=wBpOYSwtmFk155db", "start": 7, "end": 18},
        {"url": "https://youtu.be/4xq3BsKTzYo?si=PowzQRkc9UJHBijH", "start": 43, "end": 47}
        
    ],
    "deadlift": [
        #all manual screen record clips
        
    ],
    "dumbbell_biceps_curls": [
        {"url": "https://youtube.com/shorts/2jpteC44QKg?si=SgofrLdWC9lla_Wm", "start": 0, "end": 8},
        {"url": "https://youtube.com/shorts/803JIAWBj_c?si=7K1JcqsVHM0_PJwv", "start": 3, "end": 14},
        {"url": "https://youtube.com/shorts/MCC0Wj9RErI?si=3KREoLtzXVXWeFWS", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/2dfDQGXvSYo?si=556wgAS7vyuLfC4W", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/iui51E31sX8?si=_6IVDuKkef9We8PF", "start": 7, "end": 9}
        
    ],
    
    "overhead_trap_raises": [
        {"url": "https://youtube.com/shorts/JadOGd042P0?si=L_IXT2YibTLVoV-O", "start": 0, "end": 26},
        {"url": "https://youtube.com/shorts/vrR2nIKOH2s?si=84wEbkNbUkdr3h5Y", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/KbK6ByIg7bw?si=eDj0tQeke0bT7_yd", "start": 2, "end": 7}
        
    ],
    "side_lateral_raise": [
        {"url": "https://youtube.com/shorts/8EROITjumZM?si=LrdVrIG9aGNfzHI8", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/G-piLwLu0d4?si=X8N4weZpvrqSZc92", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/R-bxVwJEhFE?si=LR9FcYpE4XLzJWLm", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/rXz1mOrbQm8?si=CuwvGHzBCioRY9-L", "start": 1, "end": 9}
        
    ],

    # --- CUSTOM CHEST ---
    "wall_pushup": [
        {"url": "https://youtube.com/shorts/8GTtX8gC-lU?si=YVjoGMthE8Gno8b7", "start": 2, "end": 7},
        {"url": "https://youtube.com/shorts/-TMXETQfnRU?si=xYPYrKb-485cGjPU", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/-TMXETQfnRU?si=ryxLUmk5IRXTjPOc", "start": 5, "end": 7},
        {"url": "https://youtube.com/shorts/-TMXETQfnRU?si=nRVkyTn-qsEmCzgD", "start": 10, "end": 12},
        {"url": "https://youtube.com/shorts/Bei1RbZdFgI?si=WrGPk7ZNax_ZJfFO", "start": 0, "end": 7},
        {"url": "https://youtu.be/5NPvv40gd3Q?si=eWiNCZu_GiVs3XDh", "start": 30, "end": 36},
        {"url": "https://youtu.be/5NPvv40gd3Q?si=eWiNCZu_GiVs3XDh", "start": 41, "end": 46},
        {"url": "https://youtu.be/5NPvv40gd3Q?si=eWiNCZu_GiVs3XDh", "start": 46, "end": 51},
        {"url": "https://youtu.be/5NPvv40gd3Q?si=eWiNCZu_GiVs3XDh", "start": 52, "end": 58},
        {"url": "https://youtu.be/IBmBttR_HVI?si=ScpJhFHYquw1zxEd", "start": 106, "end": 112}

    ],
    "dumbbell_floor_press": [
        {"url": "https://youtu.be/AqYFvc9t_vU?si=aeUooqLOiVDtHIVY", "start": 29, "end": 35},
        {"url": "https://youtu.be/AqYFvc9t_vU?si=cW_xaybL6hVBIaWW", "start": 48, "end": 51},
        {"url": "https://youtu.be/AqYFvc9t_vU?si=i9_ajA1Mx9vyZbwZ", "start": 100, "end": 109},
        {"url": "https://youtu.be/AqYFvc9t_vU?si=r6SKBnXTEqgFQQlu", "start": 132, "end": 138},
        {"url": "https://youtube.com/shorts/Nk3497uvOjE?si=7NbDmDzhtvbDjgJW", "start": 0, "end": 17}
        
    ],
    "decline_pushup": [
        {"url": "https://youtu.be/5QFjmotLfW4?si=nQ-DOK3-segmDAJx", "start": 14, "end": 23},
        {"url": "https://youtube.com/shorts/jBH1xj-wDjk?si=KGs1jnJemFcUpWZv", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/dcV-ATSeryA?si=RM2vx3tisPvzgUA7", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/NKRl7wLXtyk?si=Orh-rancBQLHo4yT", "start": 0, "end": 5},
        {"url": "https://youtu.be/gP4In_xJijY?si=5hohjjlhXYz990t6", "start": 68, "end": 74},
        {"url": "https://youtu.be/gP4In_xJijY?si=5hohjjlhXYz990t6", "start": 102, "end": 105},
        {"url": "https://youtube.com/shorts/Bei1RbZdFgI?si=xQGuQ45c4O0GtMPR", "start": 0, "end": 7}

    ],
    "dumbbell_floor_flye": [
        {"url": "https://youtu.be/Z2plyYaToHQ?si=n2UX_iksNs_Fw6we", "start": 45, "end": 50},
        {"url": "https://youtu.be/Z2plyYaToHQ?si=Y0udP8ONwVGSVCiB", "start": 94, "end": 97},
        {"url": "https://youtu.be/Jz7oEmzhnfE?si=s9qnsx7q-YBLz-JS", "start": 19, "end": 25},
        {"url": "https://youtu.be/Jz7oEmzhnfE?si=H-BFKQVvv_aF-hla", "start": 40, "end": 45},
        {"url": "https://youtube.com/shorts/vld1VNiQ6xo?si=jBa4VEO3MZqKP5VJ", "start": 0, "end": 2}

    ],

    # --- CUSTOM BACK ---
    "pullup": [
        {"url": "https://youtube.com/shorts/znJVDmsSB9M?si=KR0yzExU8CNpwfGx", "start": 11, "end": 14},
        {"url": "https://youtube.com/shorts/hIGGPD2KrFo?si=Q-OA5siKN2wY-GJn", "start": 0, "end": 4},

    ],
    "renegade_row": [
        {"url": "https://www.youtube.com/watch?v=qG9wWKEgSX0", "start": 24, "end": 27},
        {"url": "https://www.youtube.com/watch?v=qG9wWKEgSX0", "start": 28, "end": 34},
        {"url": "https://www.youtube.com/watch?v=4qEIChzM4ZA", "start": 180, "end": 187},
        {"url": "https://www.youtube.com/watch?v=4qEIChzM4ZA", "start": 213, "end": 218},
        {"url": "https://youtube.com/shorts/iOh2-ySYJCI?si=7TGxiJudh16mR5Zg", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/gLLRE-em_1s?si=x2Tt6bCPKuvOv5QH", "start": 5, "end": 7},
        {"url": "https://youtube.com/shorts/gLLRE-em_1s?si=x2Tt6bCPKuvOv5QH", "start": 16, "end": 18},
        {"url": "https://youtube.com/shorts/gLLRE-em_1s?si=x2Tt6bCPKuvOv5QH", "start": 26, "end": 30}

    ],
    "good_morning": [
        {"url": "https://www.youtube.com/watch?v=5DonmXxz6Qk", "start": 10, "end": 16},
        {"url": "https://youtube.com/shorts/ZkPMJDff3do?si=supXoLXWJJWiAxfB", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/ZkPMJDff3do?si=supXoLXWJJWiAxfB", "start": 6, "end": 9},
        {"url": "https://youtube.com/shorts/qYRInOnF_HE?si=LE7sLMIIJclVPoZ_", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/qYRInOnF_HE?si=LE7sLMIIJclVPoZ_", "start": 2, "end": 4}
        
    ],

    # --- CUSTOM TRICEPS ---
    "standing_overhead_extension": [
        {"url": "https://youtube.com/shorts/kuBA3SXTNp4?si=Zx8wVPWDQuBDjctY", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/N5ncWOjvj0I?si=i2U8rqhJXjt3Px7u", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/jsQnhVg47rA?si=GweoLYU8yrs3bvCJ", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/xiXJxlGKryY?si=_kvnI56guloSlUO2", "start": 0, "end": 4}


    ],
    "dumbbell_kickback": [
        {"url": "https://youtube.com/shorts/GZ3NzlJs_yg?si=_NS5iSB5qi8Wc-zy", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/XJV627mCN3s?si=t7epde7IOB6PJzP-", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/w0oo0y_I2FM?si=HBFJbjJhSFpObn-U", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/vatMoowhfaE?si=uagnMdU8IKYTMVYG", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/_TnoxdFzwYw?si=wYABISwAl4rXWtrX", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/ZGjHc9NnJ-4?si=ukaQ6I0YXV4Zzn2c", "start": 0, "end": 5}

    ],
    "floor_skull_crusher": [
        {"url": "https://youtube.com/shorts/WixZQEPpQgw?si=fXiF3tqsuAqZsx0d", "start": 0, "end": 14},
        {"url": "https://youtube.com/shorts/7q0Uiat5vS8?si=zvr3VqkpwAJYdYce", "start": 0, "end": 13},
        {"url": "https://youtube.com/shorts/YRKjhLPLXho?si=7-kIEjcbfDcYhjum", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/7Ob8tuSaAwE?si=g0xJoS9QcUZPTbg-", "start": 0, "end": 12},
        {"url": "https://youtube.com/shorts/LswN1OwwvKU?si=bHpR0XFYOvFCIeo2", "start": 0, "end": 7}
    ],
    "plank_triceps_extension": [
        {"url": "https://www.youtube.com/watch?v=ACyLmO9wj98", "start": 85, "end": 96},
        {"url": "https://youtu.be/WDIpL0pjun0?si=ZvVzLxBskAmA1Tfs", "start": 0, "end": 12}

    ],

    # --- CUSTOM BICEPS ---
    "reverse_curl": [
        {"url": "https://youtube.com/shorts/lSjJBjeD_0I?si=JulLoFO2xV5rs5om", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/Dd1wdWUqIGo?si=f9HPk9XU9dKvPpKz", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/9dprVMhZXtI?si=PyD_82m985iAWSyU", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/09AYfVFf7pg?si=qLf07lZYVpabNlbb", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/Bh52XZDF_zM?si=ElaCKmHJVepeu1x0", "start": 3, "end": 7},
        {"url": "https://www.youtube.com/watch?v=C-mF7PAQgDM", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/RFMwylBpTjw?si=8Scs223X_uHB-k4-", "start": 0, "end": 17}

    ],
    "concentration_curl": [
        {"url": "https://youtube.com/shorts/DhxxngpUEV4?si=JYAFO4Vd-UN6RvGY", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/s7x-DTAsCwU?si=4ZR6vExIm9hxJHa_", "start": 0, "end": 5}
        
    ],

    # --- CUSTOM CORE & ABS ---
    "sit_up": [
        {"url": "https://youtube.com/shorts/hgjSJN3SXC4?si=X3WicnGj8UC6wzot", "start": 10, "end": 18},
        {"url": "https://youtube.com/shorts/acUeRZOQrQ4?si=xVEc5-W1l1GHw6fX", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/tcvgFyGEF2o?si=-g2vYXpGie_Dbr7k", "start": 1, "end": 5},
        {"url": "https://youtube.com/shorts/KjMrNdO6XA0?si=RpCtziwb9-FXap-y", "start": 6, "end": 8},
        {"url": "https://youtube.com/shorts/qXpYgvQ6_m4?si=REssqjFkggJtiwUo", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/8EyfKu-tYGs?si=eQAhnsMexzn04RR3", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/aBP3x1N4Kjw?si=95uMK5HTuME1bfsv", "start": 0, "end": 2}

    ],
    "seated_knee_tucks": [
        {"url": "https://youtu.be/7emz03H7-Bo?si=yONl6wW1H6nRMqRz", "start": 34, "end": 49},
        {"url": "https://youtube.com/shorts/N0sxmQx_GXU?si=rKRmUjnUTLJdV5tA", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/c8Z8S8YL8zo?si=yHfRAiIR4QNeqLqR", "start": 0, "end": 8},
        {"url": "https://youtube.com/shorts/V0-KQCD1ENQ?si=M1kDqX-31hfohvaG", "start": 0, "end": 9},
        {"url": "https://youtube.com/shorts/XoEQ1DDaCe0?si=JlvODv-KoFprlPeY", "start": 0, "end": 4}

    ],
    "lying_leg_raise": [
        {"url": "https://youtu.be/Wp4BlxcFTkE?si=fPEJfeLSVKpF6IDQ", "start": 15, "end": 28},
        {"url": "https://youtube.com/shorts/TEFg7ncPi8A?si=0eHY5trA5nwuSAQf", "start": 0, "end": 9},
        {"url": "https://youtu.be/dGKbTKLnym4?si=JStL4FZ_RFG0H-Q2", "start": 9, "end": 16},
        {"url": "https://youtu.be/dGKbTKLnym4?si=JStL4FZ_RFG0H-Q2", "start": 17, "end": 28},
        {"url": "https://youtube.com/shorts/vzWX0Sn2-98?si=kgH_GBVhf3L89N6n", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/2roDBvtUIm0?si=K5_g0aI4ZO8SvQ1A", "start": 18, "end": 36}
    ],
    "russian_twist": [
        {"url": "https://youtube.com/shorts/hLAZ-IgVXPw?si=LTXdEnxM2mKb1elH", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/_d5cpAre7_w?si=ykUR3a2dCrHHvQ6Y", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/I7z_2H848wE?si=vRRyOdFl8ucwQYJ5", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/WR7hlaEYdEE?si=22waIDQR_8gc9UXh", "start": 0, "end": 3},
        {"url": "https://youtube.com/shorts/LOZz5fI2TWI?si=DNwnZyt-gqKAjq3g", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/Y8C0EasD4PA?si=537PPZJEaTDu2NHR", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/GqVIYynuQy4?si=UbpZird49iqRrzW3", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/iFQV6q4xRXM?si=9DQMpw5KitYd2Q4d", "start": 0, "end": 2},
        {"url": "https://youtube.com/shorts/ykkKI_0g9jc?si=dfToiwaTXN00LhAE", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/Am0RxfZXz1w?si=JqkcLc0GatrUiZoD", "start": 0, "end": 2}
    ],
    "tricep_dips": [
        {"url": "https://youtube.com/shorts/3Ex02P8t2JA?si=OLJPOh-YgabCoQUu", "start": 0, "end": 9},
        {"url": "https://youtube.com/shorts/7l7jNfdcPn8?si=Hd5IbPRkYUrzxT8p", "start": 3, "end": 11},
        {"url": "https://youtube.com/shorts/7l7jNfdcPn8?si=Hd5IbPRkYUrzxT8p", "start": 11, "end": 20},
        {"url": "https://youtube.com/shorts/7l7jNfdcPn8?si=Hd5IbPRkYUrzxT8p", "start": 20, "end": 28},
        {"url": "https://youtu.be/PjVF-FF8JkU?si=h5JNqEvk94sAxLno", "start": 54, "end": 65},
        {"url": "https://youtu.be/aCa7cc8ECp8?si=sC21PY4n1M4-PPLy", "start": 65, "end": 78}
        
    ]
}

def download_bad_videos():
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    
    # Save to a completely separate folder to isolate bad form data
    output_dir = "raw_bad_videos"
    os.makedirs(output_dir, exist_ok=True)
    
    for exercise_name, videos in BAD_FORM_DATASET.items():
        if not videos:
            continue
            
        print(f"\nDownloading {len(videos)} BAD FORM clips for {exercise_name.upper()}...")
        
        for i, video in enumerate(videos):
            url = video.get("url")
            start = video.get("start")
            end = video.get("end")
            
            if not url or start is None or end is None:
                print(f"⚠️ Skipping invalid entry in {exercise_name}: {video}")
                continue
            
            # Explicitly label the output file as bad form
            out_file = os.path.join(output_dir, f"{exercise_name}_bad_{i+1}.mp4")
            
            if os.path.exists(out_file):
                print(f"Skipping {out_file} (Already exists)")
                continue
                
            print(f"[{i+1}/{len(videos)}] Downloading {url} (Sec {start} to {end})...")
            
            cmd = [
                sys.executable, "-m", "yt_dlp",
                "-f", "bv*[ext=mp4]/b[ext=mp4]/best",
                "--download-sections", f"*{start}-{end}",
                "--ffmpeg-location", ffmpeg_path,
                "-o", out_file,
                url
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                print(f"✅ Extracted!")
            else:
                print(f"⚠️ Failed to download clip. Video might be private or link is dead.")
    
if __name__ == "__main__":
    download_bad_videos()