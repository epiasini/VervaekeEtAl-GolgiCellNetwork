'''
Generate n_trials different Golgi network with the Vervaeke2010 algorithm, and save cell positions and edge lists to disk as csv files.
'''
import time
import random
import csv

if __name__ == "__main__":
    from java.lang import System, Float, Long
    from java.io import File
    from java.util import Vector, ArrayList
    from ucl.physiol import neuroconstruct as nc

    n_trials = 100

    pm = nc.project.ProjectManager(None,None)
    project_path = '../VervaekeEtAl-GolgiCellNetwork.ncx'
    project_file = File(project_path)
    project = pm.loadProject(project_file)

    sim_config_name = 'Default Simulation Configuration'
    sim_config = project.simConfigInfo.getSimConfig(sim_config_name)

    # create csv writer objects. Results of trials after the first are
    # appended at the end.
    cell_positions_file = open('cell_positions.csv', 'wb')
    edge_lists_file = open('edge_lists.csv', 'wb')
    cell_positions_writer = csv.writer(cell_positions_file)
    edge_lists_writer = csv.writer(edge_lists_file)

    for trial in range(n_trials):
        nC_seed = Long(random.getrandbits(32))
        # generate network
        pm.doGenerate(sim_config_name, nC_seed)
        while pm.isGenerating():
            time.sleep(0.02)
        print('network number %d generated' % trial)
        # extract connectivity structure
        conn_names = ['NetConn_CellGroup_1_CellGroup_1']
        syn_conns = [project.generatedNetworkConnections.getSynapticConnections(conn) for conn in conn_names]
        edges = []
        for conn in syn_conns:
            edges.extend([(sc.sourceEndPoint.cellNumber, sc.targetEndPoint.cellNumber) for sc in conn if sc.props[0].weight])
        # extract cell positions
        cell_positions = [(pos_record.x_pos, pos_record.y_pos, pos_record.z_pos) for pos_record in project.generatedCellPositions.getPositionRecords('CellGroup_1')]
        # save to disk
        cell_positions_writer.writerows(cell_positions)
        edge_lists_writer.writerows(edges)

    System.exit(0)


