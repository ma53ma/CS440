# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush
import queue

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    q = queue.Queue()
    q.put(maze.getStart())
    prevDict = {maze.getStart() : None}
    visitDict = {maze.getStart() : 'v'}
    path = [];

    while not q.empty():
        curr = q.get()
        neighbors = maze.getNeighbors(curr[0], curr[1], curr[2])
        for neighbor in neighbors:
            if maze.isValidMove(neighbor[0],neighbor[1], neighbor[2]):
                if neighbor not in visitDict:
                    prevDict[neighbor] = curr
                    if maze.isObjective(neighbor[0], neighbor[1], neighbor[2]):
                        path.append(neighbor)
                        currBack = prevDict[neighbor]
                        while currBack is not None:
                            path.append(currBack)
                            currBack = prevDict[currBack]
                        path.reverse()
                        return path
                    visitDict[neighbor] = 'v'
                    q.put(neighbor)
    return None
