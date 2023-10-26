#Supposizioni fatte:
#Gli ID prodotti dalla re-identificazione sono ordinati rispetto al vettore di bb teste e corpi (identities[4] corrisponde all persona in bodyBB[4] e headBB[4])
#Le BB sono in formato COCO (Angolo in alto a sinistra e dimensioni) [x_min, y_min, width, height]
#TO_DO:

#de = DistanceEstimator()
#ids = [2,4,6,8,0,3,5,7,9]
#bbb = [[5,5,2,2],[5,5,4,4],[5,5,6,6],[5,5,8,8],[5,5,0,0],[5,5,3,3],[5,5,5,5],[5,5,7,7],[5,5,9,9]]
#hbb = [[6,5,1,1],[6,5,2,2],[6,5,3,3],[6,5,4,4],[6,5,5,5],[6,5,6,6],[6,5,7,7],[6,5,8,8],[6,5,10,10]]
#idsq = [2,5,9]
#bbbq = [[10,10,]]
#text = de.computeDistance(ids,bbb,hbb)
#print(de.idsCenters)
#print("-----")
#print(de.idsDistances)
#print("-----")
#print(text)

class DistanceEstimator():
    
    def __init__(self):

        self.pxToMmRatio = [1,1,1,1]

        #DICTIONARIES PARAMETER INITIALIZATION
        self.idsCenters = {}
        self.idsDistances = {}

        for i in range (0,100):
            self.idsCenters[i] = []
            self.idsDistances[i] = []



    def computeDistance(self, identities, bodyBB, headBB):
        
        messages = []

        for i in range (len(identities)):

            center = (bodyBB[i][0] + bodyBB[i][2]) / 2
            print(identities[i])
            self.idsCenters[int(identities[i])].append(center)

            headWidth  = (headBB[i][2]- headBB[i][0])* self.pxToMmRatio[0]  #headBB[i][2] * self.pxToMmRatio[0]
            headHeight = (headBB[i][3]- headBB[i][1])* self.pxToMmRatio[1]  #headBB[i][3] * self.pxToMmRatio[1]
            bodyWidth  = (bodyBB[i][2]- bodyBB[i][0])* self.pxToMmRatio[2]  #bodyBB[i][2] * self.pxToMmRatio[2]
            bodyHeight = (bodyBB[i][3]- bodyBB[i][1])* self.pxToMmRatio[3]  #bodyBB[i][3] * self.pxToMmRatio[3]

            if abs(1 - ((headWidth * headHeight) / (bodyWidth * bodyHeight))) > 0.5:
                distance = headWidth * 0.6 + headHeight * 0.4
            else:
                distance = headWidth * 0.4 + headHeight * 0.2 + bodyWidth * 0.2 + bodyHeight * 0.2

            self.idsDistances[int(identities[i])].append(distance)

            message = "Distance: " + str(distance)

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
            message = message + "not enought sample"
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