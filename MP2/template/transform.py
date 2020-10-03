
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
# 
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.
    
        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, ,z,r)] of goals
            obstacles (list): [(x, y, z,r)] of obstacles
            window (tuple): (width, height,length) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    numLims = Arm.getNumArmLinks(arm)
    alpha_lims = (0,0)
    beta_lims = (0,0)
    gamma_lims = (0,0)
    limits = Arm.getArmLimit(arm)
    #print("limits",limits)
    #print(len(limits))
    for i in range(len(limits)):
        if i == 0:
            alpha_lims = limits[i]
        elif i == 1:
            beta_lims = limits[i]
        elif i == 2:
            gamma_lims = limits[i]
    #print("gamma",gamma_lims)
    alpha = int((alpha_lims[1] - alpha_lims[0])/granularity) + 1
    beta = int((beta_lims[1] - beta_lims[0])/granularity) + 1
    gamma = int((gamma_lims[1] - gamma_lims[0]) / granularity) + 1
    start = Arm.getArmAngle(arm)
    if len(start) == 1:
        start = (start[0],0,0)
    elif len(start) == 2:
        start = (start[0],start[1],0)
    startIdx = angleToIdx([start[0],start[1],start[2]],[alpha_lims[0],beta_lims[0],gamma_lims[0]],granularity)
    #print(startIdx)
    map = []
    for a in range(alpha):
        beta_row = []
        for b in range(beta): #beta_row.append(SPACE_CHAR)
            gamma_row = []
            for c in range(gamma):
                gamma_row.append(SPACE_CHAR)
            beta_row.append(gamma_row)
        map.append(beta_row)
    #print(len(map),len(map[0]))
    for a in range(alpha_lims[0],alpha_lims[1] + granularity, granularity):
        for b in range(beta_lims[0],beta_lims[1] + granularity, granularity):
            for c in range(gamma_lims[0],gamma_lims[1]+granularity, granularity):
                Arm.setArmAngle(arm,(a,b,c))
                armPos = Arm.getArmPos(arm)
                armEnd = Arm.getEnd(arm)
                armPosDist = Arm.getArmPosDist(arm)
                index = angleToIdx([a,b,c],[alpha_lims[0],beta_lims[0],gamma_lims[0]],granularity)
                if map[index[0]][index[1]][index[2]] != SPACE_CHAR:
                    continue
                #print(index)
                touchObstacles = doesArmTouchObjects(armPosDist,obstacles, False)
                touchGoals = doesArmTouchObjects(armPosDist, goals, True)
                if index[0] is startIdx[0] and index[1] is startIdx[1] and index[2] is startIdx[2]:
                    map[index[0]][index[1]][index[2]] = START_CHAR #
                elif not isArmWithinWindow(armPos, window) or touchObstacles[0]:
                    map[index[0]][index[1]][index[2]] = WALL_CHAR
                    if touchObstacles[1] == armPosDist[0]:
                        for shortBeta in range(beta_lims[0],beta_lims[1] + granularity, granularity):
                            for shortGamma in range(gamma_lims[0],gamma_lims[1]+granularity, granularity):
                                shortIndex = angleToIdx([a,shortBeta,shortGamma],[alpha_lims[0],beta_lims[0],gamma_lims[0]],granularity)
                                map[shortIndex[0]][shortIndex[1]][shortIndex[2]] = WALL_CHAR
                    if len(armPosDist) >= 2 and touchObstacles[1] == armPosDist[1]:
                        for shortGamma in range(gamma_lims[0], gamma_lims[1] + granularity, granularity):
                            shortIndex = angleToIdx([a, b, shortGamma],[alpha_lims[0], beta_lims[0], gamma_lims[0]], granularity)
                            map[shortIndex[0]][shortIndex[1]][shortIndex[2]] = WALL_CHAR
                elif touchGoals[0] and not doesArmTipTouchGoals(armEnd, goals):  # goals works fine, just need to add more walls
                    map[index[0]][index[1]][index[2]]  = WALL_CHAR
                    if touchObstacles[1] == armPosDist[0]:
                        for shortBeta in range(beta_lims[0], beta_lims[1] + granularity, granularity):
                            for shortGamma in range(gamma_lims[0], gamma_lims[1] + granularity, granularity):
                                shortIndex = angleToIdx([a, shortBeta, shortGamma],
                                                        [alpha_lims[0], beta_lims[0], gamma_lims[0]], granularity)
                                map[shortIndex[0]][shortIndex[1]][shortIndex[2]] = WALL_CHAR
                    if len(armPosDist) >= 2 and touchObstacles[1] == armPosDist[1]:
                        for shortGamma in range(gamma_lims[0], gamma_lims[1] + granularity, granularity):
                            shortIndex = angleToIdx([a, b, shortGamma], [alpha_lims[0], beta_lims[0], gamma_lims[0]],
                                                    granularity)
                            map[shortIndex[0]][shortIndex[1]][shortIndex[2]] = WALL_CHAR
                elif touchGoals[0] and doesArmTipTouchGoals(armEnd, goals):
                    map[index[0]][index[1]][index[2]] = OBJECTIVE_CHAR
                    if touchObstacles[1] == armPosDist[0]:
                        for shortBeta in range(beta_lims[0], beta_lims[1] + granularity, granularity):
                            for shortGamma in range(gamma_lims[0], gamma_lims[1] + granularity, granularity):
                                shortIndex = angleToIdx([a, shortBeta, shortGamma],[alpha_lims[0], beta_lims[0], gamma_lims[0]], granularity)
                                map[shortIndex[0]][shortIndex[1]][shortIndex[2]] = OBJECTIVE_CHAR
                    if len(armPosDist) >= 2 and touchObstacles[1] == armPosDist[1]:
                        for shortGamma in range(gamma_lims[0], gamma_lims[1] + granularity, granularity):
                            shortIndex = angleToIdx([a, b, shortGamma], [alpha_lims[0], beta_lims[0], gamma_lims[0]],granularity)
                            map[shortIndex[0]][shortIndex[1]][shortIndex[2]] = OBJECTIVE_CHAR
    return Maze(map,[alpha_lims[0],beta_lims[0],gamma_lims[0]],granularity)