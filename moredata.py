import csv

def LoadCSV (datafile,delimiter=',') :
    lines = []
    with open (datafile) as csvfile :
        reader = csv.reader (csvfile, delimiter=delimiter)
        for line in reader :
            lines.append (line)
    return lines

ExTRA_DATA_DEBUG_COUNT=None
ExTRA_DATA_PRINT_DEBUG=False
def process_csv_lines (lines, xmin_index, ymin_index, xmax_index, ymax_index, datadir, fileindex, labelindex, ignoreFirst=False) :
    count = 0
    carlines = []
    notcarlines = []
    carbbs = []
    notcarbbs = []
    for line in lines :
        count += 1
        if 1 == count and ignoreFirst:
            continue
        bbox =(( int(line[xmin_index]), int(line[ymin_index]) ), ( int(line[xmax_index]), int(line[ymax_index]) ))
        if bbox[0][0] >= bbox[1][0] or bbox[0][1] >= bbox[1][1] :
            # Ignore invalid bounding boxes
            continue
        if line[labelindex].upper() == 'CAR' :
            carlines.append ('{}/{}'.format(datadir, line[fileindex]))
            carbbs.append (bbox)
        else :
            notcarlines.append ('{}/{}'.format(datadir, line[fileindex]))
            notcarbbs.append (bbox)
        if ExTRA_DATA_DEBUG_COUNT != None and count >= ExTRA_DATA_DEBUG_COUNT :
            break
    return carlines, carbbs, notcarlines, notcarbbs


OBJ_DET_DATA_DIR='../data/object-detection-crowdai'
OBJ_DET_LABELS_CSV='{}/{}'.format(OBJ_DET_DATA_DIR, 'labels.csv')
OBJ_DET_FILE_INDEX=4
OBJ_DET_LABEL_INDEX=5
OBJ_DET_XMIN_INDEX=0
OBJ_DET_YMIN_INDEX=1
OBJ_DET_XMAX_INDEX=2
OBJ_DET_YMAX_INDEX=3

OBJ_DATASET_DIR='../data/object-dataset'
OBJ_DATASET_LABELS_CSV='{}/{}'.format(OBJ_DATASET_DIR, 'labels.csv')
OBJ_DATA_FILE_INDEX=0
OBJ_DATA_XMIN_INDEX=1
OBJ_DATA_YMIN_INDEX=2
OBJ_DATA_XMAX_INDEX=3
OBJ_DATA_YMAX_INDEX=4
OBJ_DATA_LABEL_INDEX=6


def LoadExtraData () :
    cars = []
    car_bounding_boxes = []
    notcars = []
    notcar_bounding_boxes = []
    
    lines1 = LoadCSV (OBJ_DET_LABELS_CSV)
    cars1, carbb1, notcars1, notcarbb1 = process_csv_lines (lines1,
                                                            OBJ_DET_XMIN_INDEX, OBJ_DET_YMIN_INDEX, OBJ_DET_XMAX_INDEX, OBJ_DET_YMAX_INDEX,
                                                            OBJ_DET_DATA_DIR, OBJ_DET_FILE_INDEX, OBJ_DET_LABEL_INDEX, True)
    
    
    if ExTRA_DATA_PRINT_DEBUG :
        print ("After Set 1")
        print ("cars = {}".format (cars1))
        print ("carbbs = {}".format (carbb1))
        print ("notcars = {}".format(notcars1))
        print ("notcarbbs = {}".format (notcarbb1))
    
    lines2 = LoadCSV (OBJ_DATASET_LABELS_CSV, delimiter=' ')
    cars2, carbb2, notcars2, notcarbb2 = process_csv_lines (lines2,
                                                            OBJ_DATA_XMIN_INDEX, OBJ_DATA_YMIN_INDEX, OBJ_DATA_XMAX_INDEX, OBJ_DATA_YMAX_INDEX,
                                                            OBJ_DATASET_DIR, OBJ_DATA_FILE_INDEX, OBJ_DATA_LABEL_INDEX, False)
    
    
    if ExTRA_DATA_PRINT_DEBUG :
        print ("After Set 2")
        print ("cars = {}".format (cars2))
        print ("carbbs = {}".format (carbb2))
        print ("notcars = {}".format(notcars2))
        print ("notcars = {}".format(notcarbb2))
        #print ('od l1 file = {}, label = {}'.format (lines2[1][0], lines2[1][6]))
    
    
    cars = cars1 + cars2
    car_bounding_boxes = carbb1 + carbb2
    notcars = notcars1 + notcars2
    notcar_bounding_boxes = notcarbb1 + notcarbb2
    if ExTRA_DATA_PRINT_DEBUG :
        print ("After Combining")
        print ("Combined Cars = {}".format (cars))
        print ("Combined Carbbs = {}".format (car_bounding_boxes))
        print ("Combined Not Cars = {}".format (notcars))
        print ("Combined Not Carbbs = {}".format (notcar_bounding_boxes))
    return cars, car_bounding_boxes, notcars, notcar_bounding_boxes

LoadExtraData ()
