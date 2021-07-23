"""
This file contains a utility function to generate and modify the Abaqus
input file (*.inp) so that the results are written in the ASCII output
format (*.fil).

Author: Cristóbal Tapia Camú
E-mail: crtapia@gmail.com
"""


def get_ascii_result_file(job_obj):
    """Configure Job to save results in the ASCII format (*.fil)

    Parameters
    ----------
    job_obj : TODO

    """
    job_obj.writeInput(consistencyChecking=False)

    input_lines = ("*FILE FORMAT, ASCII\n" + "*EL FILE\n" + "S, E, COORD\n" +
                   "*NODE FILE\n" + "COORD, U\n")

    # Open *inp file
    file_name = job_obj.name + ".inp"

    with open(file_name, "r") as inp_file:
        lines = inp_file.readlines()

    with open(file_name, "w") as inp_file:
        for l in lines:
            if l == "*End Step\n":
                inp_file.writelines(input_lines)
                inp_file.write(l)
            else:
                inp_file.write(l)

    job_ascii = mdb.JobFromInputFile(
        name=job_obj.name, inputFileName=job_bsh.name + ".inp", type=ANALYSIS, memory=90,
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, explicitPrecision=SINGLE,
        nodalOutputPrecision=SINGLE, userSubroutine=job_obj.userSubroutine,
        scratch=job_obj.scratch, multiprocessingMode=DEFAULT, numCpus=job_obj.numCpus)

    return job_ascii
