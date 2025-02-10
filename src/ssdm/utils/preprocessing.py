def get_bunny_path(args):
    if args.precision_level == 'simple':
        # 500 vertices
        return "datasets/objects/manifold_example/bunny_coarse.obj"
    elif args.precision_level == 'median':
        # 2992 vertices
        return "datasets/objects/manifold_example/simplify_bunny.obj"
    elif args.precision_level == 'complex':
        # ~ 30,000 vertices (35947)
        return "datasets/objects/manifold_example/stanford_bunny.obj"
    elif args.precision_level == 'manifold':
        # ~50,000 vertices (52288)
        return "datasets/objects/manifold_example/manifold_processed_bunny.obj"
