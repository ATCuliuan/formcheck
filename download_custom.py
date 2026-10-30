import os
import sys
import subprocess
import imageio_ffmpeg

# Paste your YouTube links and timestamps inside the brackets for each exercise.
# Example format: {"url": "https://www.youtube.com/watch?v=XYZ", "start": 5, "end": 15}
CUSTOM_DATASET = {
    # CHEST
    "dumbbell_floor_press": [
        {"url": "https://youtu.be/T0Y3OBF1bNI?si=ORV0QYVW9H5hoL15", "start": 0, "end": 12},
        {"url": "https://youtu.be/lNdi7VEf2Ew?si=hgGwWVJWcmQzhIQi", "start": 65, "end": 71},
        {"url": "https://youtu.be/uUGDRwge4F8?si=ZWZGcNsgjmlWiTlX", "start": 59, "end": 87}, 
        {"url": "https://youtu.be/2NbOwF_mJ2g?si=aNHBM93Dbn7PX4xq", "start": 4, "end": 12},
        {"url": "https://youtu.be/2NbOwF_mJ2g?si=aNHBM93Dbn7PX4xq", "start": 13, "end": 24},
        {"url": "https://youtu.be/qHCI9rK7HqM?si=oRAuq3quXPmCSDG2", "start": 3, "end": 13},
        {"url": "https://youtu.be/z9pbIdHzwXw?si=EiHwg8i5g0OcW-3b", "start": 0, "end": 7},
        {"url": "https://youtu.be/oRAF41LDrW8?si=IJ_p_grWYta2foA6", "start": 35, "end": 45},
        {"url": "https://youtu.be/oRAF41LDrW8?si=IJ_p_grWYta2foA6", "start": 46, "end": 52},
        {"url": "https://youtu.be/_KwWsz38j6M?si=NYMVRBbSR3_qyuLk", "start": 8, "end": 24}
    ],
    "decline_pushup": [
        {"url": "https://youtu.be/QBlYp-EwHlo?si=_tOvCjWEJ1czZwrN", "start": 0, "end": 6},
        {"url": "https://youtu.be/SKPab2YC8BE?si=FYuAlXe4ZfM7Ryoc", "start": 56, "end": 64},
        {"url": "https://youtu.be/O7dVvwEK9J4?si=SAsYkV1JmaJFcKgZ", "start": 86, "end": 105},
        {"url": "https://youtu.be/5QFjmotLfW4?si=WR64nBPzOVWDL63z", "start": 28, "end": 32},
        {"url": "https://youtu.be/X92uGqa1eeU?si=k3AgovSsrCJkAOvZ", "start": 15, "end": 30},
        {"url": "https://youtu.be/b_CC4kAF1HQ?si=cGYcZVOKX82kgCFZ", "start": 36, "end": 47},
        {"url": "https://youtube.com/shorts/lx-49Yjqm-Q?si=9I_xL1UFV3sW-W9r", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/zZbheQN4GvE?si=YVeTLB5D7kWkchQb", "start": 0, "end": 10},
        {"url": "https://youtu.be/0V8xUxGvo4A?si=IaG0O02LWkJk97gx", "start": 0, "end": 12},
        {"url": "https://youtu.be/s5dEc154cCc?si=9txfLqRccEbCcQun", "start": 130, "end": 137},
        {"url": "https://youtu.be/OjPfLfLsw3c?si=-to5RXxfKmycheUJ", "start": 8, "end": 15}
    ],
    "dumbbell_floor_flye": [
        {"url": "https://youtu.be/bgC53-J-6gA?si=yu-3HF_zGaQQ4pcY", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/WwWg5VksY18?si=4Zmira28-obu1_lm", "start": 0, "end": 19},
        {"url": "https://youtu.be/AM6pUVHLLd8?si=lNMMoCclFBqKk8DF", "start": 1, "end": 14},
        {"url": "https://youtu.be/LPxiXYW7kmo?si=BlVStqQvYUZQCgtC", "start": 35, "end": 48},
        {"url": "https://youtube.com/shorts/zrq4P7Pll4w?si=-eedir3VITHpHl29", "start": 0, "end": 11},
        {"url": "https://youtu.be/JMoQWdn5Efg?si=ry-lYJ04BWaUMXfl", "start": 0, "end": 27},
        {"url": "https://youtube.com/shorts/Hw-RxxBIIoA?si=YwyJPRijHUIJOG1D", "start": 0, "end": 11},
        {"url": "https://youtu.be/f5EMDU6d9fY?si=GFG7T_1P4WwP-atx", "start": 4, "end": 30},
        {"url": "https://youtu.be/tBsDcptmalY?si=LLdofS7nkpQAvB6r", "start": 2, "end": 20},
        {"url": "https://youtube.com/shorts/_B62sCCpEgQ?si=GQvuW-0brvvCIZOJ", "start": 43, "end": 53}
    ],

    # BACK
    "renegade_row": [
        {"url": "https://youtu.be/Q28cLuweLv4?si=55B5KpiTSLcc-lXY", "start": 0, "end": 12},
        {"url": "https://youtube.com/shorts/kcQp0AmYeug?si=zrAPW0ssYWsFQJ9K", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/BrDRJdHquzk?si=thS_z93sChQbR1Pb", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/Knzg7ru3dsE?si=r_hdzTVvZGKr9CWg", "start": 2, "end": 15},
        {"url": "https://youtu.be/OlpPaGyywyY?si=2h3069j5cxM1trxf", "start": 54, "end": 60},
        {"url": "https://youtu.be/gi1EwrDbgJo?si=lGarZgDGNcouieJC", "start": 0, "end": 5},
        {"url": "https://youtu.be/SUqsBTSldyk?si=TRauSdVLI3iGpjGS", "start": 0, "end": 21},
        {"url": "https://youtu.be/cwQ60ewHMpk?si=eDsqSuOEzia_GLEY", "start": 0, "end": 20},
        {"url": "https://youtube.com/shorts/85WI7XdSzSw?si=2h0jHuDM5JXO2rXo", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/uQ5AQ6k4f-Y?si=ioKNQph45kQNNY_Q", "start": 1, "end": 15}
    ],
    "good_morning": [
        {"url": "https://youtube.com/shorts/uQ5AQ6k4f-Y?si=ioKNQph45kQNNY_Q", "start": 6, "end": 24},
        {"url": "https://youtu.be/61zbhuRiwQg?si=I1UReYuUAVyQMdOT", "start": 11, "end": 17},
        {"url": "https://youtu.be/fQ2olCTc0-c?si=zx4JMZID3h4sp_y9", "start": 24, "end": 34},
        {"url": "https://youtu.be/nWyx81AfTos?si=QOLnfWFLO4kyljOf", "start": 0, "end": 13},
        {"url": "https://youtu.be/5Xj6XUa77qc?si=CVgXzUEZBGi9lvaP", "start": 33, "end": 43},
        {"url": "https://youtube.com/shorts/sAAn3y-qxP4?si=NlUbmDLRZfRl-Yrh", "start": 11, "end": 16},
        {"url": "https://youtu.be/7mrKMteISXs?si=QpU5CtjrzOLGB3kP", "start": 11, "end": 23},
        {"url": "https://youtu.be/8ivUt9iXMuA?si=Mnngh8ialCXj5Ptc", "start": 0, "end": 42},
        {"url": "https://youtube.com/shorts/tNLDpesr1IE?si=r6AdgI1qqxh9NBbA", "start": 0, "end": 7},
        {"url": "https://youtube.com/shorts/9prcXIdXev0?si=HNF9_yBMrAsPdQEj", "start": 7, "end": 18},
        {"url": "https://youtube.com/shorts/H1yq11eeSas?si=rnJkxfbSHkBDGEqd", "start": 2, "end": 6}
    ],

    # TRICEPS
    "standing_overhead_extension": [
        {"url": "https://youtube.com/shorts/QKfRhY8qvhg?si=n9FyzEd_g5udiVm_", "start": 0, "end": 10},
        {"url": "https://youtu.be/4--u52sHZPs?si=Xi_jzIbY73kLOAeo", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/vS5pWxLmtUw?si=VpQEYK9VsCxbZdE2", "start": 30, "end": 44},
        {"url": "https://youtube.com/shorts/GVqVmi56dvA?si=Pi9bw3jhxMJZG-jd", "start": 0, "end": 14},
        {"url": "https://youtu.be/7h3lG2WnLXg?si=-3wxZMPS2sgb4a78", "start": 0, "end": 8},
        {"url": "https://youtu.be/TtnFe9HVp70?si=AwrH308tSpZvB3st", "start": 44, "end": 85},
        {"url": "https://youtube.com/shorts/SYb_-Os9xX8?si=RhHsLqNKB5RWOP-_", "start": 0, "end": 20},
        {"url": "https://youtube.com/shorts/SO4gja7-kxg?si=UM62rgqg8KJRUIde", "start": 0, "end": 8},
        {"url": "https://youtube.com/shorts/plUVYa4epzc?si=6WyBY_CHHSvVSMKB", "start": 0, "end": 7},
        {"url": "https://youtube.com/shorts/6I8v2frZTIg?si=uOkByLskct1j0jyX", "start": 20, "end": 35}
    ],
    "dumbbell_kickback": [
        {"url": "https://youtu.be/m_UlDFNX4mk?si=WQgwz5T-KeW1zP60", "start": 0, "end": 7},
        {"url": "https://youtu.be/ShCYaoHmWmk?si=eM1KkUvTR38kjusY", "start": 14, "end": 23},
        {"url": "https://youtu.be/f3E7eEq2c6c?si=3hMSdCe3z4kjs4oI", "start": 13, "end": 23},
        {"url": "https://youtu.be/noKoND5ghI4?si=D9Aj6zzwQjvs7Kr7", "start": 12, "end": 31},
        {"url": "https://youtu.be/KzBZ02EAJvE?si=B4mQKST6iYYab4uT", "start": 120, "end": 164},
        {"url": "https://youtu.be/TtnFe9HVp70?si=FMrDVQzEAiymvkYt", "start": 104, "end": 149},
        {"url": "https://youtube.com/shorts/y_K39SFu1J0?si=QjxyHD-yA30edxsM", "start": 5, "end": 15},
        {"url": "https://youtube.com/shorts/uCIHDiZiG04?si=r51BKcC9-gIxoa4H", "start": 10, "end": 26},
        {"url": "https://youtube.com/shorts/Tr8TVqPq3AU?si=7PTIbMrVjc7R1ciu", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/PsGUdSyUoLw?si=oJmX-HrAW4iJjaMO", "start": 0, "end": 7}
    ],
    "floor_skull_crusher": [
        {"url": "https://youtube.com/shorts/YRKjhLPLXho?si=518tTxCXDl_VjznO", "start": 0, "end": 10},
        {"url": "https://youtube.com/shorts/2sZkbsLub38?si=Vs4XKeQosagrE5BG", "start": 27, "end": 46},
        {"url": "https://youtu.be/Py4I0J6i2kY?si=md7dtM3qc0Kf0Zwd", "start": 2, "end": 32},
        {"url": "https://youtu.be/XuvPA_KtSTo?si=d84jHI0R-nPVpmnC", "start": 79, "end": 86},
        {"url": "https://youtu.be/v5KK23dYzH8?si=wvYlL8Y6RDGI-ARi", "start": 5, "end": 17},
        {"url": "https://youtu.be/1BDGIcMTSXc?si=g-QV_wI4zyxv1FGY", "start": 0, "end": 9},
        {"url": "https://youtube.com/shorts/BJINoCQe5-k?si=Ix4RZHCEylW5n-TO", "start": 0, "end": 17},
        {"url": "https://youtube.com/shorts/EVhBBVYd_gQ?si=ZS0YMlySq7ZrC2hg", "start": 0, "end": 8},
        {"url": "https://youtu.be/IQsE_Zi6dU4?si=1ctFSZgmrk5ur51w", "start": 0, "end": 14},
        {"url": "https://youtube.com/shorts/DB-fd1wZNUk?si=f3frOlHZmUFznfFX", "start": 0, "end": 11}
    ],
    "plank_triceps_extension": [
        {"url": "https://youtu.be/1bjwi_z72sI?si=xYb7wpswpXW0TAi-", "start": 0, "end": 14},
        {"url": "https://youtube.com/shorts/Q0JjS_mi7LQ?si=4FX2sfDxQvnHDMD2", "start": 0, "end": 7},
        {"url": "https://youtu.be/5u1aCdsky8M?si=mwhGGNr_9ZCA5x7L", "start": 0, "end": 12},
        {"url": "https://youtube.com/shorts/62c2Bj9n3JY?si=hnZNZChu6oh-GrV_", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/T6XgoVj5h-g?si=6MWXsTqdzTX3p_rm", "start": 10, "end": 36},
        {"url": "https://youtube.com/shorts/Xlpdw_KQux4?si=klvRIYrQ7tOl8kbQ", "start": 16, "end": 22},
        {"url": "https://youtube.com/shorts/QC3_UAaHJ8w?si=RC5o5RUgWvk5q-UX", "start": 7, "end": 18},
        {"url": "https://youtu.be/O5hgu-Eu8Yo?si=ZCfu8g-uD4oVMMt7", "start": 0, "end": 11},
        {"url": "https://youtu.be/-nfTC718a1I?si=ssJ1cVxERlnHtY7g", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/gJtPsPGTRIs?si=JVH6btLnr2x5wEos", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/S7cxffRVXZ8?si=1UA4vfAPbupaWUAS", "start": 0, "end": 16},
        {"url": "https://youtube.com/shorts/WBlirPzh9j8?si=1k1yAHYZoroPCj0p", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/gsuBGTJE_ls?si=3rZsOWkeVgYlkOMS", "start": 5, "end": 13},
        {"url": "https://youtu.be/BTIFXsPiY44?si=NYLYhbzpi6jl7Hhd", "start": 0, "end": 14},
        {"url": "https://youtube.com/shorts/FaWUdVysHWE?si=XnESkiDrf2njPv9g", "start": 0, "end": 8},
    ],

    # BICEPS
    "reverse_curl": [
        {"url": "https://youtu.be/hUA-fIpM7nA?si=I7ocLnIKZHjiiujS", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/59EuFkx9akU?si=zXxjPd-cBV4HmDeS", "start": 0, "end": 5},
        {"url": "https://youtu.be/vM40o2TiJfM?si=YeplqlWQ4SLZ1EKe", "start": 2, "end": 11},
        {"url": "https://youtube.com/shorts/ItO5j6DXS9Y?si=fz8CmRZAmBTgEye1", "start": 0, "end": 8},
        {"url": "https://youtu.be/EoYWMB856cA?si=QDLB0QvVTMmlnM_h", "start": 12, "end": 23},
        {"url": "https://youtu.be/EoYWMB856cA?si=QDLB0QvVTMmlnM_h", "start": 28, "end": 38},
        {"url": "https://youtube.com/shorts/4biCTQej6RQ?si=huuOJgtUd3qpIQhI", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/1Nvg_vHUJPY?si=Nm6FsnlxivoSMIYN", "start": 6, "end": 11},
        {"url": "https://youtu.be/rV2asdIdXF0?si=rJdF97u-7tYXaqDf", "start": 109, "end": 119},
        {"url": "https://youtu.be/CIGPx1bx0Hs?si=439p0TS2q95SFwQ0", "start": 7, "end": 27},
        {"url": "https://youtube.com/shorts/IRKizybC5sk?si=da5012mxD6c53W8T", "start": 5, "end": 15},
        {"url": "https://youtube.com/shorts/MkbwtkpvYew?si=UeSoRJivkgnS7n23", "start": 0, "end": 23},
        {"url": "https://youtu.be/rV2asdIdXF0?si=rJdF97u-7tYXaqDf", "start": 109, "end": 119},
        {"url": "https://youtube.com/shorts/fB-8ikmGOJE?si=prpmC8nxdUxcKZeV", "start": 0, "end": 7}
    ],
    "concentration_curl": [
        {"url": "https://youtu.be/VMbDQ8PZazY?si=pnHMcn9SoLxjehwi", "start": 1, "end": 26},
        {"url": "https://youtu.be/Xc47YLxFftQ?si=3ehn0dY95i937svU", "start": 3, "end": 19},
        {"url": "https://youtube.com/shorts/toN9a_yHf5Q?si=B1Oc0zAoO33P1GWk", "start": 0, "end": 12},
        {"url": "https://youtu.be/dgvKbfP6lTU?si=fU4zqRBFvgHo9wDR", "start": 48, "end": 70},
        {"url": "https://youtu.be/ZcU2hN76UyA?si=i9Xhl5lhO2OYkmj1", "start": 47, "end": 53},
        {"url": "https://youtube.com/shorts/esYZR2kT8BY?si=f0EoPAennhfd5fvZ", "start": 4, "end": 13},
        {"url": "https://youtu.be/nMIGb5-ytRE?si=ErMwQcxSaI7fisZr", "start": 4, "end": 14},
        {"url": "https://youtube.com/shorts/IqCunCwr9kI?si=Kmxov-rqUkD-w_r3", "start": 0, "end": 10},
        {"url": "https://youtu.be/Jvj2wV0vOYU?si=iryhTW_PIGoQpy9R", "start": 57, "end": 69},
        {"url": "https://youtube.com/shorts/YEgmqqn9obo?si=xxKp9jpwTBZTsFWO", "start": 22, "end": 37}
    ],

    # CORE & ABS
    "seated_knee_tucks": [
        {"url": "https://youtu.be/HR_1AqBmVAU?si=wG1XX6dYd6T9lHLY", "start": 2, "end": 18},
        {"url": "https://youtube.com/shorts/x2umFUuBLRw?si=XG4xTr1itP8wsuFa", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/zKSNRAkSeJQ?si=TmsJXGG6TRmLvecR", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/FP-H-Sdk4RA?si=-ICUw4r-96OHtPH0", "start": 0, "end": 4},
        {"url": "https://youtube.com/shorts/BhAEvKRomE0?si=Obw4qg-4Pr1RaIN-", "start": 0, "end": 8},
        {"url": "https://youtu.be/d_up4yEZhFI?si=PFTbD6zSTwfDq3Pl", "start": 11, "end": 35},
        {"url": "https://youtu.be/t1etqgnWrHo?si=ac3pF8fBaQliyVBu", "start": 12, "end": 21},
        {"url": "https://youtube.com/shorts/UmhwN3ps0j8?si=6TwIs20ftH-nE4Cu", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/PNOXh26m0-Q?si=qvbG2M3sCYyUFWBx", "start": 0, "end": 12},
        {"url": "https://youtu.be/1HyXlzqrJEs?si=yy1B_3Pf9BnbQAV6", "start": 3, "end": 10},
        {"url": "https://youtube.com/shorts/_7mlIoS80y4?si=SHakXlBfyERfX-3x", "start": 0, "end": 9},
        {"url": "https://youtu.be/7emz03H7-Bo?si=m0NBXbQ2W0ewm7dG", "start": 22, "end": 32},
        {"url": "https://youtube.com/shorts/MNzNeH--Pgs?si=ixnU5v4o1NA9vdrQ", "start": 0, "end": 7},
        {"url": "https://youtu.be/19Jgv1-EwJw?si=iCZGyuwOkgF_GRYA", "start": 2, "end": 12},
        {"url": "https://youtu.be/EZIg8zkf_2M?si=ltH4bXY2N0bwAWdO", "start": 0, "end": 34}
    ],
    "lying_leg_raise": [
        {"url": "https://youtube.com/shorts/kLa7E4YhVQo?si=XJpD1sZ-Uy9F3_Uj", "start": 24, "end": 31},
        {"url": "https://youtube.com/shorts/PkLYH3uRtSs?si=7wuHrj0Mw6s8akzZ", "start": 4, "end": 9},
        {"url": "https://youtube.com/shorts/2wUpI98Ix-k?si=Vf6LgeRgyRRBG86u", "start": 8, "end": 20},
        {"url": "https://youtu.be/xJJu-WiROM8?si=NRSSZAt9U-2J7nuP", "start": 28, "end": 40},
        {"url": "https://youtube.com/shorts/TEFg7ncPi8A?si=0eHY5trA5nwuSAQf", "start": 10, "end": 20},
        {"url": "https://youtube.com/shorts/6mDUQZTdrtM?si=FL1EnERovwvkA8vk", "start": 0, "end": 6},
        {"url": "https://youtube.com/shorts/wV5CzTt5Vm0?si=7Inhl0DPr8jbRKYo", "start": 2, "end": 18},
        {"url": "https://youtube.com/shorts/FijNSgahpz0?si=wBFHpttqOSEFXAbe", "start": 0, "end": 12},
        {"url": "https://youtube.com/shorts/6mDUQZTdrtM?si=FOBNsdrceHeOtGhp", "start": 6, "end": 12}
    ],
    "russian_twist": [
        {"url": "https://youtu.be/99T1EfpMwPA?si=1k67de04JcXPZU7K", "start": 10, "end": 19},
        {"url": "https://youtu.be/PkGPokybYaU?si=q9XEYB-BTgF1Yjhb", "start": 13, "end": 24},
        {"url": "https://youtu.be/DJQGX2J4IVw?si=b8woKE7mYp1ZMLDe", "start": 15, "end": 21},
        {"url": "https://youtu.be/FShbaqrGGu4?si=IqJ9DgH4ovV0LVTC", "start": 2, "end": 8},
        {"url": "https://youtube.com/shorts/vq1FICbQC4Q?si=ZJOetJVxeq6XE4bZ", "start": 0, "end": 5},
        {"url": "https://youtube.com/shorts/hLAZ-IgVXPw?si=4mfOEg6qaRuJSbKA", "start": 4, "end": 8},
        {"url": "https://youtube.com/shorts/aRUMRbl7KS4?si=c_RUerfgnO8qJTc8", "start": 11, "end": 17},
        {"url": "https://youtube.com/shorts/p9MwoVmdt6I?si=kNAa_iwOFCpbngaL", "start": 0, "end": 11},
        {"url": "https://youtube.com/shorts/QoMna6v0QS4?si=kUWobXldA5Ird4lf", "start": 1, "end": 12},
        {"url": "https://youtube.com/shorts/FxH-nOGaPxw?si=kZgNWyVg2_syTxxy", "start": 0, "end": 6},
        {"url": "https://youtu.be/lvNPZCtt8NQ?si=-Fx1ntJ1uNERc23Y", "start": 3, "end": 8},
        {"url": "https://youtu.be/NrBUdmJSTXk?si=Or3b06eOGIJhC1VC", "start": 262, "end": 275},
        {"url": "https://youtu.be/NrBUdmJSTXk?si=Or3b06eOGIJhC1VC", "start": 277, "end": 289},
        {"url": "https://youtube.com/shorts/bfVGrSoDYFQ?si=Xpr_6_EpjlR1_05u", "start": 0, "end": 9},
        {"url": "https://youtube.com/shorts/_lOrJVh5Y-I?si=ENEJNMBIxPrHyq-H", "start": 0, "end": 12}
    ],
    "tricep_dips": [
        {"url": "https://youtube.com/shorts/ZpOOkWhByCk?si=afjbWRfCUKIOvtAl", "start": 3, "end": 10},
        {"url": "https://youtube.com/shorts/jgmQ9zj14UQ?si=lJKbMUGUoXYGhWEs", "start": 0, "end": 5},
        {"url": "https://youtu.be/qrS6aa0aQ9I?si=pi0mHD4ce5-iqbKp", "start": 18, "end": 24},
        {"url": "https://youtu.be/oUMQuLqHTkY?si=WptH9EO3VmGcqReL", "start": 17, "end": 24},
        {"url": "https://youtu.be/rSjpqBwMdIA?si=EDJY8hrgEdSBXLyX", "start": 4, "end": 25},
        {"url": "https://youtu.be/681R2RiCrnU?si=YietcTAMVQ84IhGi", "start": 0, "end": 14},
        {"url": "https://youtu.be/MDDFCrVGbOY?si=QgrXL20htFmnJm2B", "start": 0, "end": 5},
        {"url": "https://youtu.be/Q5HaswjJiHo?si=U4kuVtQs74q9vZQb", "start": 5, "end": 12},
        {"url": "https://youtu.be/xSjzOF4s4Fc?si=0OfHMh7e0_BmygKx", "start": 18, "end": 28},
        {"url": "https://youtube.com/shorts/TaqV6LmnZg0?si=JClrr9sZcKyg6C8z", "start": 0, "end": 8},
        {"url": "https://youtube.com/shorts/E90Wo668qHI?si=QIKXgHzTtbZBFXG5", "start": 0, "end": 7},
        {"url": "https://youtube.com/shorts/OqdcjoKLIvY?si=CFYhKe8wR8xF19ve", "start": 1, "end": 8}
    ]
}

def download_custom_videos():
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs("raw_videos", exist_ok=True)
    
    for exercise_name, videos in CUSTOM_DATASET.items():
        if not videos:
            continue
            
        print(f"\nDownloading {len(videos)} clips for {exercise_name.upper()}...")
        
        for i, video in enumerate(videos):
            url = video["url"]
            start = video["start"]
            end = video["end"]
            
            out_file = f"raw_videos/{exercise_name}_{i+1}.mp4"
            
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
                print(f"⚠️ Failed to download clip. Link might be dead.")

if __name__ == "__main__":
    download_custom_videos()