import jsplib_loader as jl

def main():
    for instance in jl.get_instances().values():
        model = instance.as_gurobi_balas_model(use_big_m=True)
        model.update()
        model.write(f'../jsp_mps/{instance.name}.mps')

if __name__ == "__main__":
    main()
