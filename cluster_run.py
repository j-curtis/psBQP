import pickle
import os
import sys
import time as time_lib
import numpy as np

"""
===============================================================================================================================================================================================================
Wrapper functions handling imports. Change manually depending on specifics of project.
===============================================================================================================================================================================================================
"""

# Main solver function
def import_main_solver():
    
    # Declare name of module to import
    global main_solver_function
    
    # Actual import statement
    from code_run import evolve_keldysh_state as main_solver_function
    
    # # Old import versions
    # from SCBA import solve_dyson_equation_holes as main_solver_function
    # from SCBA import solve_dyson_equation_holes_reduced as main_solver_function
    # from RPA_solver import calculate_magnon_spectra_doping as main_solver_function
    # from magnon_computation_integrated import full_magnon_calc_save_file as main_solver_function
    # from coupled_solver import debug_save_everything as main_solver_function

# Library import function
def import_library_tools():
    
    # Declare name of modules to import
    global path_management, autoprocess_tools_local
    
    # Current controlling machine, declared globally as well
    global crt_controlling_machine
    
    # Initialize path, local variable for now
    _demler_tools_path = '/cluster/work/demler/tools_library_source_code/v2.2.0-beta'
    crt_controlling_machine = 'cluster_euler'
        
    # Add path to default system addresses
    sys.path.append(_demler_tools_path)
    
    # Perform actual import
    from demler_tools.file_manager import path_management, autoprocess_tools_local
    
# Custom automatic postprocess
def import_cap_function():
    return None
#     from evolution_solver_square import evolution_tofile_1pulse as main_solver_function



"""
Arguments taken by script:
 
 - calculation_type: 	'main_simulation' is the only implemented one so far.
 			
 If 'main_simulation':
 	- input_directory: 	address where arguments should be read from
 	- point_index:		which parameter tuple to run from calculation_arguments descriptor
"""



# ===============================================================================================================================================================
# Run a single instance on given node
# ===============================================================================================================================================================

def run_1_point(input_directory, point_index, solver_function):
    
    # Retrieve arguments from binary file
    calculation_parameters_file = open(os.path.join(input_directory, 'calculation_arguments'), 'rb')
    args_array, kwargs_array = pickle.load(calculation_parameters_file)
    calculation_parameters_file.close()

    # Isolate requested arguments
    if args_array == None:
        crt_args = ()
    else:
        crt_args = args_array[point_index]

    if kwargs_array == None:
        crt_kwargs = {}
    else:
        crt_kwargs = kwargs_array[point_index]
        
    # Run the simulation
    sf_return = solver_function(*crt_args, **crt_kwargs)



# ===============================================================================================================================================================
# Interpret and run
# ===============================================================================================================================================================

if __name__ == "__main__":

    # Recover calculation type
    calculation_type = sys.argv[1]
    
    # Check whether it's main simulation
    if calculation_type == 'main_simulation':
        
        # Import of main solver function
        import_main_solver()
                
        # Recover input folder, current run index
        input_directory = sys.argv[2]
        point_index = int(sys.argv[3])
        
        # Run the actual function
        run_1_point(input_directory = input_directory, point_index = point_index, solver_function = main_solver_function)
        
    # Check whether it's automatic postprocessing
    if calculation_type == 'auto_postprocessing':
        
        # Get necessary library tools
        import_library_tools()
        
        # Recover input directory
        input_directory = sys.argv[2]
        
        # Recover other arguments for library initialization
        _lib_project_name = sys.argv[3]
        
        # Library initialization. Should pass parameters via control file here!
        path_management.initialize_minimalistic(project_name = _lib_project_name, crt_controlling_machine = crt_controlling_machine)
        
        # Run autoprocessing method
        autoprocess_tools_local.auto_postprocessor_local(folder_path = input_directory)


