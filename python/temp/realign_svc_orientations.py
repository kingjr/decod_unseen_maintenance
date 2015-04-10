"""This scripts takes the categorical predictions of a orientation SVC
    and realign them to be visually meaningful"""

def realign_svc_orientations(results):
    # Load classification results
    gat = pickle.load(open( results, "rb" ))

    # realign to 4th angle
    for a in [15, 45, 75, 105, 135, 165]
        sel = gat.y == a;
        probas(sel,:,:,:) = results.probas(:,sel,:,:,[a:6 1:(a-1)]);
