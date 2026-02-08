- [ ] Can we have the roads defined by math functions ? Like we just have a polynom or something like that, this way it is super fast to work with it. We can even have this during real deployment were we try to fit some polynom to the road given the camera and work with that polynom.

---

> So autonomous driving is a supervised learning task, tesla has all the data, they just need to annotate it all and the network would receive the current state and predict the action to take, no reinforcement learning is required in here ? And same thing for controlling a robotic arm to learn to transfer objects between bins, we don't need any reinforcement learning do we ?

Indeed, they first train classical neural networks to predict what and how to control, but on top of that they add reinforcement learning through **RLHF** (Reinforcement Learning From Human Feedback) to handle out of distribution scenarios.
