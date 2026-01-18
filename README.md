# SATcore-Designer
A constraint-based chip layout designing tool using the z3 SAT solver to automatically place components while respecting adjacency and distance requirements between components.

## Overview

SATcore-Designer allows you to design chip layouts by defining components and constraints in XML format and then using SAT the z3 SAT solver to find a valid placement automatically.

## Features

- **Component Placement**: Automatically positions rectangular components of up to 19 types on a chip canvas
- **Adjacency Constraints**: Enforce that specific components must be adjacent to eachother or all others
- **Distance Constraints**: Define minimum distances between component instances

## Getting Started

### Prerequisites

- Python 3.12.3
- Dependencies listed in the pyproject.toml file

### Usage

1. Create a virtual environment with poetry install and source it 
2. Define your chip layout in an XML file (see example.xml for format and usage)
3. Run the designer in the terminal to generate a valid placement or an unsat

## Project Structure

- 'src': Main SAT-based layout solver
- 'example.xml': Sample xml file
- 'earlier files/': Jupyter notebooks for experimentation and tutorials
