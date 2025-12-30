# Improved-Dijkstra
This version of Dijkstra algorithm mainly focuses on feasibility on real-world machines. It employs base Dijkstra but also adds penalties if near to obstacle and sharp turns. These addition makes the algorithm real-world friendly.

The code works in following fashion:
1) Obstacle Inflation - Obstacles are inflated using distance transform. This avoids planning paths not close to obstacles. This feature keeps real-world tolerances in mind.
2) Dijkstra Algorithm - This section is traditional Dijkstra Algorithm. There are very little changes and additional penalties are added to the path formation.
3) Bézier Smoothing - This section is post processing of the path. We employ Bézier smoothing for this process. The point where the path has sharp turn is smoothed down. This feature mainly focuses for applications in real-world.
4) Environment Generation
5) Visualization - We mainly use MatPlotLib for visualizing obstacles and generation on path.

The first version - dijkstra_simple.py is basically same code but simple environment. Here the code focuses on safety and might follow wall-following behaviour.
The second version - dijkstra_complex.py has complex environment. The code is slightly modified and algorithm tries not to show wall-following behaviour.
The main motive for the algorithm is to make it real-world friendly and not just impressive on simulations.
Main feature in this code is that we can tailor it according to the robot or rover we are using provided we have its URDF file.
