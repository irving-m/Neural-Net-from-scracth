net = Network(x, y, [Layer(np.array([[0.7, -0.3, -0.1],
                                [2, 1.5, -0.5]]), 
                        np.array([1.2, .3]),
                        ReLu, d_ReLu
                        ), 
                    Layer(np.array([[1.2, -0.6]]),
                          np.array([1]),
                          ReLu, d_ReLu
                        )
                    ]
                )