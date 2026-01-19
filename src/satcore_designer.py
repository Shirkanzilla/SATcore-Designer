import argparse
from z3 import *
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys

def process_arguments():
    parser = argparse.ArgumentParser(prog='satcore_designer.py')
    parser.add_argument("constraint_file", help="xml file containing the constraints for the chip.\n\
                    Adhere to the layed out in example.xml\n\
                    ")
    parser.add_argument("--save_path", "-s", help="Path where the image of the chip is saved")
    return parser.parse_args()

def process_xml(path: str) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    try:
        xml = ET.parse(path).getroot()
    except:
        sys.exit(1)

    chip_size = (int(xml[0].text), int(xml[1].text))

    components = []
    xmlComponents = xml.find("components")
    for idx, comp in enumerate(xmlComponents.iter("component")):
        amount = int(comp[0].text)
        width = int(comp[1].text)
        height = int(comp[2].text)
        for i in range(amount):
            components.append([width, height, idx])
    
    adjacency = []
    xmlAdjacencies = xml.find("adjacencies")
    for idx, adj in enumerate(xmlAdjacencies.iter("adjacency")):
        comp1 = adj[0].text
        comp2 = adj[1].text
        if comp1 == "*":
            adjacency.append([comp1, int(comp2)-1])
        elif comp2 == "*":
            adjacency.append([int(comp1)-1, comp2])
        else:
            adjacency.append([int(comp1)-1, int(comp2)-1])

    distance = []
    xmlDistances = xml.find("distances")
    for idx, dist in enumerate(xmlDistances.iter("distance")):
        comp1 = int(dist[0].text)
        comp2 = int(dist[1].text)
        value = int(dist[2].text)
        distance.append([comp1 - 1, comp2 - 1, value])
    
    return chip_size, components, adjacency, distance

 # Overlap in one axis
def is_overlapping(pos1, size1, pos2, size2):
    return And(
        pos1 < pos2 + size2,
        pos2 < pos1 + size1
    )

# Checks if one axis is overlapping and if so, checks whether the other axis is neighboring
def is_adjacent(comp_x, comp_y, comp_width, comp_height, comp2_x, comp2_y, comp2_width, comp2_height):
    is_horizontally_overlapping = is_overlapping(comp_x, comp_width, comp2_x, comp2_width)
    is_horizontally_adjacent = And(
        is_horizontally_overlapping,
        Or(
            comp2_y + comp2_height == comp_y,
            comp_y + comp_height == comp2_y
        )
    )

    is_vertically_overlapping = is_overlapping(comp_y, comp_height, comp2_y, comp2_height)
    is_vertically_adjacent = And(
        is_vertically_overlapping,
        Or(
            comp2_x + comp2_width == comp_x,
            comp_x + comp_width == comp2_x
        )
    )

    return Or(is_horizontally_adjacent, is_vertically_adjacent)


def build_chip_constraints(chip_size: tuple, components: NDArray, adjacancy: NDArray, distance: NDArray) -> (NDArray, NDArray):
    s = Solver()
    xyb = []
    for idx, component_element in enumerate(components):
        comp_x = Int(f"comp{idx}_x")
        comp_y = Int(f"comp{idx}_y")
        comp_is_vertical = Bool(f"comp{idx}_is_vertical")
        s.add(comp_x >= 0, comp_y >= 0)
        s.add(
            If
            (
                comp_is_vertical, 
                And
                (
                    comp_x + component_element[0] <= chip_size[0], 
                    comp_y + component_element[1] <= chip_size[1]
                ),
                And
                (
                    comp_x + component_element[1] <= chip_size[0], 
                    comp_y + component_element[0] <= chip_size[1]
                )
            )
        )
        xyb.append([comp_x, comp_y, comp_is_vertical, component_element[2]])
    xyb = np.array(xyb)

    for idx, (comp_x, comp_y, comp_is_vertical, _) in enumerate(xyb):
            for idx2, (comp2_x, comp2_y, comp2_is_vertical, _) in enumerate(xyb[idx+1:], start=idx+1):
                s.add(
                    Or
                    (
                        comp_x + If(comp_is_vertical, components[idx][0], components[idx][1]) <= comp2_x,
                        comp2_x + If(comp2_is_vertical, components[idx2][0], components[idx2][1]) <= comp_x,
                        comp_y + If(comp_is_vertical, components[idx][1], components[idx][0]) <= comp2_y,
                        comp2_y + If(comp2_is_vertical, components[idx2][1], components[idx2][0]) <= comp_y
                    )
                )

    # adjacancy constraints:
    for (comp1, comp2) in adjacancy:
        components1 = []
        components2 = []
        if comp1 == "*":
            components2 = xyb[xyb[:,3] == comp2]
            components1 = xyb[np.invert(np.isin(xyb[:,3], components2[:,3]))]
        elif comp2 == "*":
            components2 = xyb[xyb[:,3] == comp1]
            components1 = xyb[np.invert(np.isin(xyb[:,3], components2[:,3]))]    
        else:
            components1 = xyb[xyb[:,3] == comp1]
            components2 = xyb[xyb[:,3] == comp2]
        for comp_x, comp_y, comp_is_vertical, comp_id in components1:
            idx = np.where(xyb[:,3] == comp_id)[0][0]
            comp_width = If(comp_is_vertical, components[idx][0], components[idx][1])
            comp_height = If(comp_is_vertical, components[idx][1], components[idx][0])
            clause = False
            for comp2_x, comp2_y, comp2_is_vertical, comp2_id in components2:
                idx2 = np.where(xyb[:,3] == comp2_id)[0][0]
                comp2_width = If(comp2_is_vertical, components[idx2][0], components[idx2][1])
                comp2_height = If(comp2_is_vertical, components[idx2][1], components[idx2][0])
                clause = Or(clause, is_adjacent(comp_x, comp_y, comp_width, comp_height, comp2_x, comp2_y, comp2_width, comp2_height))
            s.add(clause)

    # distance constraints:
    for (comp1, comp2, value) in distance:
        components1 = xyb[xyb[:,3] == comp1]
        components2 = xyb[xyb[:,3] == comp2]
        for comp_x, comp_y, comp_is_vertical, comp_id in components1:
            idx = np.where(xyb[:,3] == comp_id)[0][0]
            comp_width = If(comp_is_vertical, components[idx][0], components[idx][1])
            comp_height = If(comp_is_vertical, components[idx][1], components[idx][0])
            comp_center_x = (comp_x + comp_width / 2) - 0.5
            comp_center_y = (comp_y + comp_height / 2) - 0.5
            for comp2_x, comp2_y, comp2_is_vertical, comp2_id in components2:
                if comp_x == comp2_x and comp_y == comp2_y:
                    continue
                idx2 = np.where(xyb[:,3] == comp2_id)[0][0]
                comp2_width = If(comp2_is_vertical, components[idx2][0], components[idx2][1])
                comp2_height = If(comp2_is_vertical, components[idx2][1], components[idx2][0])
                comp2_center_x = (comp2_x + comp2_width / 2) - 0.5
                comp2_center_y = (comp2_y + comp2_height / 2) - 0.5
                dx = comp_center_x - comp2_center_x
                dy = comp_center_y - comp2_center_y
                s.add(Or(
                        dx * dx >= value * value,
                        dy * dy >= value * value,
                    ))

    if s.check() == sat:
        return xyb, s.model()
    else:
        return xyb, None
    
def translate_model_to_image(xyb: NDArray, chip_size: tuple, components: NDArray, model: ModelRef) -> NDArray:
    image_grid = np.zeros(chip_size)
    # color in the rest of the components
    for idx, (comp_x, comp_y, comp_is_vertical, comp_id) in enumerate(xyb):
        x = model[comp_x].as_long()
        y = model[comp_y].as_long()

        if model[comp_is_vertical]:
            image_grid[x : x + components[idx][0], y : y + components[idx][1]] = comp_id + 1
        else:
            image_grid[x : x + components[idx][1], y : y + components[idx][0]] = comp_id + 1
    return image_grid

def map_key2scale(key, n):
    return (key+0.5) * (n)/(n+1)

def visualize_result(xyb: NDArray, chip_size: tuple, components: NDArray, model: ModelRef, save_path: str) -> None:
    n = max(np.array(components)[:,2]) + 1

    if n < 12:
        cmap = plt.get_cmap("Paired", n+1)
    else:
        cmap = plt.get_cmap("tab20", n+1)

    labels = {
        0: "Unassigned",
    }

    for i in range(n):
        labels[i+1] = f"Component {i+1}"
    
    #for i in range(max(components[:,3])
    image_grid = translate_model_to_image(xyb, chip_size, components, model).T

    fig, ax = plt.subplots()
    im = ax.imshow(image_grid, cmap=cmap, interpolation="nearest", origin="upper", aspect="equal")
    ax.set_xticks(np.arange(-0.5, chip_size[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, chip_size[1], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1)

    cbar = fig.colorbar(im, ax=ax, ticks=[map_key2scale(key, n) for key in labels.keys()])
    cbar.ax.set_yticklabels(list(labels.values()))
    #plt.show()
    plt.savefig(save_path)

if __name__ == "__main__":
    args = process_arguments()
    chip_size, components, adjacency, distance = process_xml(args.constraint_file)
    print(distance)
    print("processing...")
    xyb, model = build_chip_constraints(chip_size, components, adjacency, distance)
    if (model == None):
        print("Constraints are not satisfiable.")
        sys.exit(1)
    else:
        print("Chip designed successfully!")
        save_path = "chip"
        if args.save_path:
            save_path = args.save_path
        visualize_result(xyb, chip_size, components, model, save_path)
        sys.exit(0)