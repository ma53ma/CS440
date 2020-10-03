# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *
from numpy import linalg as la
import copy


def computeCoordinate(start, length, angle):
    x_add = int(length*math.cos(math.radians(angle)))
    y_add = int(length*math.sin(math.radians(angle)))
    return start[0] + x_add,start[1] - y_add

def eucDist(one, two):
    return np.sqrt(np.square(one[0] - two[0]) + np.square(one[1] - two[1]))

def doesArmTouchObjects(armPosDist, objects, isGoal):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
                list of tuples?
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """
    #need to cite this before turning it in
    # http://geomalgorithms.com/a02-_lines.html
    #print(armPosDist)
    touchingLink = 0
    for link in armPosDist:
        linkStart = link[0]
        linkEnd = link[1]
        linkSeg = (linkEnd[0] - linkStart[0],linkEnd[1] - linkStart[1]) # P1 - P0
        for object in objects:
            v = (object[0] - linkStart[0],object[1] - linkStart[1]) # P - P0 - starting pt
            dp1 = np.dot(v,linkSeg)
            dp2 = np.dot(linkSeg,linkSeg)
            if dp1 <= 0:
                dist = eucDist(object,linkStart)# start pt
            elif dp2 <= dp1:
                dist = eucDist(object, linkEnd) # end pt
            else:
                x = link[0][0] + (dp1/dp2)*linkSeg[0],link[0][1]+(dp1/dp2)*linkSeg[1]
                dist = eucDist(object,x)
            if isGoal:
                if dist <= (object[2]):
                    if link == armPosDist[0]:
                        touchingLink = 1
                    elif link == armPosDist[1]:
                        touchingLink = 2
                    elif link == armPosDist[2]:
                        touchingLink = 3
                    return True, touchingLink
            else:
                if dist <= (object[2] + link[2]):
                    if link == armPosDist[0]:
                        touchingLink = 1
                    elif link == armPosDist[1]:
                        touchingLink = 2
                    elif link == armPosDist[2]:
                        touchingLink = 3
                    return True, touchingLink
    return False, touchingLink

def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tick touches any goal. False if not.
    """
    for goal in goals:
        if eucDist(armEnd,(goal[0],goal[1])) <= goal[2]:
            return True
    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Can just check two joints to see
        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """
    #print(armPos)
    for link in armPos:
        if link[1][0] > window[0] or link[1][0] < 0 or link[1][1] > window[1] or link[1][1] < 0 :
            return False
    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
