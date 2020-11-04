# Coded Distributed Computing

This repository is going to be a big monorepo for my study on coded distributed computing.
The reason for choosing a monorepo instead of many smaller repos is entirely because of the nature of using 
Numpy and Scipy on Windows machines in combination with Visual Studio Code. There is a need to carefully setup 
a Visual Studio Code workspace so that code execution works properly since with Windows I need to use Miniconda 
to install Numpy and Scipy. 

Hence to minimise repeated efforts copying and pasting around Visual Studio Code workspace configurations from 
one task to the next everything is going to go in a big monorepo. 

So far the analysis that this repository contains includes:

- Singular Value Decomposition

## Singular Value Decomposition

A repository for the study of the Singular Value Decomposition.

### Work In Progress

For the time being I'm going to focus on looking at the error characteristics of Matrix-Vector multiplications of low rank approximations provided by the singular-value decomposition.

Related Obsidian Note: LibraryOfKevin/Graceful Degradation of the Performance of Matrix Operations to Meet Timing Constraints