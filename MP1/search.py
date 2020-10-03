# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

import queue
import heapq
import copy
"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    q = queue.Queue()
    q.put(maze.getStart())
    prevDict = {maze.getStart() : None}
    visitDict = {maze.getStart() : 'v'}
    path = [];

    while not q.empty():
        curr = q.get()
        neighbors = maze.getNeighbors(curr[0], curr[1])
        for neighbor in neighbors:
            if maze.isValidMove(neighbor[0],neighbor[1]):
                if neighbor not in visitDict:
                    prevDict[neighbor] = curr
                    if maze.isObjective(neighbor[0], neighbor[1]):
                        path.append(neighbor)
                        currBack = prevDict[neighbor]
                        while currBack is not None:
                            path.append(currBack)
                            currBack = prevDict[currBack]
                        path.reverse()
                        return path
                    visitDict[neighbor] = 'v'
                    q.put(neighbor)

class Node:

    g = 0
    h = 0
    prev = None
    location = (0,0)
    obj = []

    def __init__(self, prev, location):
        self.prev = prev
        self.location = location

    def __eq__(self, other):
        return self.location == other.location and str(self.obj) == str(other.obj)

    def __lt__(self, other):
        return self.g + self.h < other.g + other.h

    def __le__(self, other):
        return self.g + self.h <= other.g + other.h

    def __gt__(self, other):
        return self.g + self.h > other.g + other.h

    def __ge__(self, other):
        return self.g + self.h >= other.g + other.h

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    ends = maze.getObjectives()
    endpt = ends[0]
    return single_astar(maze, maze.getStart(), endpt)

def single_astar(maze, start_p, end):
    if start_p[0] is end[0] and start_p[1] is end[1]:
        return []
    start = Node(None,start_p)
    heap = []
    start.h = manDist(start.location, end)
    heapq.heappush(heap, start)
    visited = {start.location: 'v'}
    path = []
    while len(heap) != 0:
        curr = heapq.heappop(heap)
        neighbors = maze.getNeighbors(curr.location[0], curr.location[1])
        for neighbor in neighbors:
            neighborNode = Node(curr, neighbor)
            if neighborNode.location not in visited and neighborNode not in heap:
                if neighbor == end: # this seems like the problem
                    path.append(neighbor)
                    currBack = neighborNode.prev
                    while currBack is not None:
                        path.append(currBack.location)
                        currBack = currBack.prev
                    path.reverse()
                    return path
                neighborNode.g = neighborNode.prev.g + 1
                neighborNode.h = manDist(neighbor,end)
                visited[neighborNode.location] = 'v'
                heapq.heappush(heap, neighborNode)

def shortest_multi_astar(maze):
    ends = maze.getObjectives()
    ends.sort()
    edgeList = {}
    for endOne in ends:
        for endTwo in ends:
            edgeList[(endOne, endTwo)] = len(single_astar(maze, endOne, endTwo))
    heap = []
    path = []
    distHeap = []
    start = Node(None, maze.getStart())
    start.obj = ends
    mstCost = {}  # do i need an initial value
    visited = {(start.location, str(start.obj)): 'v'}
    cost = {(start.location, str(start.obj)): 0}
    heapq.heappush(heap, start)
    mstCost[str(start.obj)] = MST(maze, start.obj, edgeList)
    while len(heap) != 0:
        currNode = heapq.heappop(heap)
        for neighbor in maze.getNeighbors(currNode.location[0], currNode.location[1]):
            neighborNode = Node(currNode, neighbor)
            neighborNode.obj = copy.deepcopy(currNode.obj)
            if neighbor in currNode.obj:  # maybe make currPt lose its end instead of neighbor
                neighborNode.obj.remove(neighbor)
                neighborNode.obj.sort()
                if len(neighborNode.obj) == 0:
                    path.append(neighbor)
                    currBack = currNode
                    currNode.obj.append(neighbor)  # now has that last objective
                    while currBack is not None:
                        path.append(currBack.location)  # add 4,1
                        currBack = currBack.prev
                        # print(path)
                    path.reverse()
                    return path
                if str(neighborNode.obj) not in mstCost:
                    mstCost[str(neighborNode.obj)] = MST(maze, neighborNode.obj, edgeList)
            for end in currNode.obj:  # not totally sure if this is ok
                # print("neighbor is : " + str(neighbor))
                # print("potential end is " + str(end))
                heapq.heappush(distHeap, (manDist(neighborNode.location, end), end))
            closestDot = heapq.heappop(distHeap)
            distHeap.clear()
            pot_g = currNode.g + 1
            pot_h = heuristic(mstCost, neighborNode, closestDot[1])
            comparisonNode = Node(None, neighbor)
            comparisonNode.obj = copy.deepcopy(currNode.obj)
            if (neighborNode.location, str(neighborNode.obj)) not in visited:
                neighborNode.g = pot_g
                neighborNode.h = pot_h
                cost[(neighbor, str(neighborNode.obj))] = pot_g + pot_h
                heapq.heappush(heap, neighborNode)
                visited[(neighborNode.location, str(neighborNode.obj))] = 'v'
                # print("pushing " + str(neighborNode.location) + "," + str(neighborNode.obj) + "prev: " + str(neighborNode.prev.location))
                # print("pushing " + str(neighbor) + " with " + str(ends))
def manDist(one, two):
    return abs(one[0] - two[0]) + abs(one[1] - two[1])

def heuristic(mstCost, neighborNode, dot):
    return mstCost[str(neighborNode.obj)] + manDist(neighborNode.location, dot)


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    return shortest_multi_astar(maze)

def MST(maze, objectives, edgeList):
    """
    :param maze:
    :return:
    """
    ends = objectives
    prevList = {}
    start = ends[0]
    cost = {start: 0}
    heap = []
    heapq.heappush(heap,(cost[start],start))
    for endptOne in ends:
        endptOneNode = Node(None, endptOne)
        if endptOne is not start:
            cost[endptOne] = float('inf') # setting distance to nodes to infinity
        heapq.heappush(heap,(cost[endptOne],endptOne)) # pushing all of these nodes onto the heap
                #
                # manDist(endptOne,endptTwo)
    for x in range(0, len(ends)): # for all nodes
        curr = heapq.heappop(heap) # popping one with least cost
        for endpt in ends:
            if curr[1] is endpt: # if we would be comparing the same node, or if the node has already been expanded, continue
                continue
            if edgeList[(curr[1], endpt)] < cost[endpt]: #if cost of current plus path cost from curr to endpt
                cost[endpt] = edgeList[(curr[1], endpt)] # replace cost
                prevList[endpt] = curr[1]
                heap.sort() #resort the heap in case anything changed
    sum = 0
    for key in prevList:
        #print(str(key) + " and " + str(prevList[key]) + "with weight" + str(edgeList[(key,prevList[key])]))
        sum = sum + edgeList[(key,prevList[key])]

    #print("sum is: " + str(sum))
    return sum

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return shortest_multi_astar(maze)
def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    # TODO: Write your code here
    ends = maze.getObjectives()
    ends.sort()
    edgeList = {}
    for endOne in ends:
        for endTwo in ends:
            edgeList[(endOne, endTwo)] = len(single_astar(maze, endOne, endTwo))
    heap = []
    path = []
    distHeap = []
    start = Node(None, maze.getStart())
    start.obj = ends
    mstCost = {}  # do i need an initial value
    visited = {(start.location, str(start.obj)): 'v'}
    cost = {(start.location, str(start.obj)): 0}
    heapq.heappush(heap, start)
    mstCost[str(start.obj)] = MST(maze, start.obj, edgeList)
    while len(heap) != 0:
        currNode = heapq.heappop(heap)
        for neighbor in maze.getNeighbors(currNode.location[0], currNode.location[1]):
            neighborNode = Node(currNode, neighbor)
            neighborNode.obj = copy.deepcopy(currNode.obj)
            if neighbor in currNode.obj:  # maybe make currPt lose its end instead of neighbor
                neighborNode.obj.remove(neighbor)
                neighborNode.obj.sort()
                if len(neighborNode.obj) == 0:
                    path.append(neighbor)
                    currBack = currNode
                    currNode.obj.append(neighbor)  # now has that last objective
                    while currBack is not None:
                        path.append(currBack.location)  # add 4,1
                        currBack = currBack.prev
                        # print(path)
                    path.reverse()
                    return path
                if str(neighborNode.obj) not in mstCost:
                    mstCost[str(neighborNode.obj)] = MST(maze, neighborNode.obj, edgeList)
            for end in currNode.obj:  # not totally sure if this is ok
                # print("neighbor is : " + str(neighbor))
                # print("potential end is " + str(end))
                heapq.heappush(distHeap, (manDist(neighborNode.location, end), end))
            closestDot = heapq.heappop(distHeap)
            distHeap.clear()
            pot_g = currNode.g + 1
            pot_h = 3*heuristic(mstCost, neighborNode, closestDot[1])
            comparisonNode = Node(None, neighbor)
            comparisonNode.obj = copy.deepcopy(currNode.obj)
            if (neighborNode.location, str(neighborNode.obj)) not in visited:
                neighborNode.g = pot_g
                neighborNode.h = pot_h
                cost[(neighbor, str(neighborNode.obj))] = pot_g + pot_h
                heapq.heappush(heap, neighborNode)
                visited[(neighborNode.location, str(neighborNode.obj))] = 'v'
                # print("pushing " + str(neighborNode.location) + "," + str(neighborNode.obj) + "prev: " + str(neighborNode.prev.location))
                # print("pushing " + str(neighbor) + " with " + str(ends))
