# Laptime_Sim

In no particular order: 

TODO's
- Add low speed/ high slip accommodations to tire model
- Add longitudinal forces
  - Brake and diff effects included
- Add overturning moment (low priority)
- Create a better plotting package/module for everything
- improve trimmed lateral acceleration determination
  - potentially use interpolation
- Save the sweep data for further analysis
- Create a more robust, post sweep data analysis package
  - Potential (google hyperparameter sweep/search)
- Develop GUI for all this
- Clean up and modularize code for use in a laptime sim
- lap time sim
- VALIDATE VALIDATE VALIDATE
- CORRELATE CORRELATE CORRELATE 
  - (especially with driver feedback)
- Explore optimization strategies to reduce computation time
  - this could mean using optimization packages over loops/convergence
  - use of different interpreters (ie. Pypy, Cython)
  - Kinematic table interpolation instead of fitted equations
  - replace lists with tuples where possible (faster)
  - remove '.' and import methods directly when possible
- Machine learning/setup optimization tools/strategy integration
- Major project restructure/refactor
- Add other analysis tools as described in milliken and Patton 
- Find a better/lighter way to store and use vehicle model
- refactor for readability
- Takle existing TODO's throughout code
- Create unit tests for modules 
- Update documentation, especially this README