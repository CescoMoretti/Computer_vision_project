import math

class DistanceEstimator():
    
    def __init__(self):

        self.realDimensions = [17,22,40,170]
        self.focalLenght = 11.0

        #DICTIONARIES PARAMETER INITIALIZATION
        self.idsCenters = {}
        self.idsDistances = {}

        for i in range (0,100):
            self.idsCenters[i] = []
            self.idsDistances[i] = []



    def computeDistance(self, identities, bodyBB, headBB):
        
        messages = []

        for i in range (len(identities)):

            message = "ID: " + str(identities[i])

            center = (bodyBB[i][0] + bodyBB[i][2]) / 2
            print(identities[i])
            self.idsCenters[int(identities[i])].append(center)

            headWidth = (self.focalLenght * self.realDimensions[0]) / (headBB[i][2] - headBB[i][0])
            headHeight = (self.focalLenght * self.realDimensions[1]) /  (headBB[i][3] - headBB[i][1])
            bodyWidth = (self.focalLenght * self.realDimensions[2]) / (bodyBB[i][2] - bodyBB[i][0])
            bodyHeight = (self.focalLenght * self.realDimensions[3]) / (bodyBB[i][3] - bodyBB[i][1])

            if abs(1 - ((headWidth * headHeight) / (bodyWidth * bodyHeight))) > 0.5:
                distance = headWidth * 0.7 + headHeight * 0.3
            else:
                distance = headWidth * 0.6 + headHeight * 0.2 + bodyWidth * 0.1 + bodyHeight * 0.1

            self.idsDistances[int(identities[i])].append(distance)

            message = message + " - Distance: " + str(math.ceil(distance*100)/100)

            direction = self.estimateDirection(int(identities[i]))
            if direction != "":
                message = message + " - Direction: " + direction

            messages.append(message)
        
        return messages

        

    def estimateDirection(self, id):

        message = ""
        
        centers = self.idsCenters[id]
        distances = self.idsDistances[id]
        
        if len(centers) < 4:
            message = message + "still not enought sample"
            return message
            
        centers = centers[-4:]
        distances = distances[-4:]

        if centers[0] < centers[2] and centers[0] < centers[3] and centers[1] < centers[2] and centers[1] < centers[3]: 
            message = message + "right "
        if centers[0] > centers[2] and centers[0] > centers[3] and centers[1] > centers[2] and centers[1] > centers[3]:
            message = message + "left "

        if distances[0] < distances[2] and distances[0] < distances[3] and distances[1] < distances[2] and distances[1] < distances[3]:
            message = message + "closer"
        if distances[0] > distances[2] and distances[0] > distances[3] and distances[1] > distances[2] and distances[1] > distances[3]:
            message = message + "farthest"

        return message
