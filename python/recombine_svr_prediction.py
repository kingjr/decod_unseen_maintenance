def recombine_svr_prediction(path_x,path_y):
    import cmath

    #load first regressor (cosine)
    gat = pickle.load(open( path_x, "rb" ) )

    x = np.array(gat.y_pred_)

    #load second regressor (sine)
    gat = pickle.load(open( path_y, "rb" ))
    y = np.array(gat.y_pred_)

    # cartesian 2 polar transformation
    angle, radius = cmath.polar(complex(x, y))

    return angle, radius
