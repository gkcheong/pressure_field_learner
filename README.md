# pressure_field_learner
Attempts to learn the Lagrangian constant of SCFT through machine learning

The code uses Pytorch to build a simple FC neural network to learn the Lagrangian
constant needed for the solution of SCFT (see: PSCF) in diblock copolymer system.

This is a work in progress:-
The model currently can learn the entire validation set but does not perform well 
for validation set.  

Things to implement:  
-k-fold cross validation  
-regularization  
-\(test) ADAM  
-\(test) conv net  
