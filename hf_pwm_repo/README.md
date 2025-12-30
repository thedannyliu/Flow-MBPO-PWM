---
license: mit
---

# PWM: Policy Learning with Large World Models

[Ignat Georgiev](https://www.imgeorgiev.com/), [Varun Giridhar](https://www.linkedin.com/in/varun-giridhar-463947146/), [Nicklas Hansen](https://www.nicklashansen.com/), [Animesh Garg](https://animesh.garg.tech/)

[Project website](http://imgeorgiev.com/pwm)  [Paper](TODO)  [Models & Datasets](https://huggingface.co/imgeorgiev/pwm)

## Overview

![](https://github.com/imgeorgiev/pwm/figures/teaser.png)

Instead of building world models into algorithms, we propose using large-scale multi-task world models as
differentiable simulators for policy learning. When well-regularized, these models enable efficient policy
learning with first-order gradient optimization. This allows PWM to learn to solve 80 tasks in < 10 minutes
each without the need for expensive online planning.

## Structure of repository

```
pwm
├── dflex
│   ├── data - data used for dflex world model pre-training
│   └── pretrained - already trained world models that can be used in dflex experiments
├── multitask - pre-trained world models for multitask evaluation
├── pedagogical - pre-trained world models for recreating pedagogical examples
└── README.md
```


