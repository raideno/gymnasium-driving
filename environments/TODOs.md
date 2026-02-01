# Road Todos

- [ ] Simplify the road implementation by using a graph based definition. The renderer will then draw everything correctly.
- [ ] Implement the observation wrappers.
- [ ] Implement some action "discretizer" wrappers.
- [ ] Implement some reward wrappers.

Checkout https://github.com/danielriege/tinycarlo/tree/main/mapbuilder which is an alternative that is pretty well made. It might be of interest.

So the idea is to define the road as a graph:
- Each node will have a position.
- Edges connecting the nodes will have some data to encode:
  - Road length.
  - Type of connection: straight line, arc, merge, etc.
  - Number of lanes

But this might be too much, so I think we'll stick to our current roads for now and teach the networks to behave on them instead.

---

- [x] Have all the dimensions be to scale, replicate parking lot, renault zoe dimensions, etc.
- [ ] Implement some control algorithms.
  - [ ] Rule based.
  - [ ] Discretized Q-Learning.
  - [ ] Policy Gradient.
  - [ ] Imitation Learning. https://www.youtube.com/watch?v=GDmhrAHxgQE
  - [ ] Neural Network Based Controllers, etc.
  - [x] Pure Pursuit.
  - [x] Stanley.
  - [x] PID.
  - [x] Clothoids.
- [x] Add real timing, fps and a controllable speed up of time.
- [x] Add collision with road sides. Enabled overall and controllable on road network on creation on each individual segment.
- [ ] Add "Distance" to lane center into the observation data.

- [ ] Ask what is the speed used for the car for test drives.
- [ ] Understand and simplify the clothoid based controller.
- [ ] Make the clothoid based controller with other types of functions such as polynoms, etc.
- [ ] Make the environment more realistic by returning predictions in the observation state, like possibility to add a model that'll take the observation and return a prediction.
