import networkx as nx
import osmium as o
import argparse
from pathlib import Path


class OSM2RNHandler(o.SimpleHandler):

    def __init__(self, rn):
        super(OSM2RNHandler, self).__init__()
        self.candi_highway_types = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified',
                                    'residential', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
                                    'tertiary_link', 'living_street', 'service', 'road'}
        self.rn = rn
        self.eid = 0

    def way(self, w):
        if 'highway' in w.tags and w.tags['highway'] in self.candi_highway_types:
            raw_eid = w.id
            full_coords = []
            for n in w.nodes:
                full_coords.append((n.lon, n.lat))
            if 'oneway' in w.tags:
                if w.tags['oneway'] != 'yes':
                    full_coords.reverse()
                for i in range(len(full_coords)-1):
                    coords = [full_coords[i], full_coords[i + 1]]
                    edge_attr = {'eid': self.eid, 'coords': coords, 'raw_eid': raw_eid, 'highway': w.tags['highway']}
                    rn.add_edge(coords[0], coords[-1], **edge_attr)
                    self.eid += 1
            else:
                for i in range(len(full_coords)-1):
                    coords = [full_coords[i], full_coords[i + 1]]
                    # add edges for both directions
                    edge_attr = {'eid': self.eid, 'coords': coords, 'raw_eid': raw_eid, 'highway': w.tags['highway']}
                    rn.add_edge(coords[0], coords[-1], **edge_attr)
                    self.eid += 1

                reversed_full_coords = full_coords.copy()
                reversed_full_coords.reverse()
                for i in range(len(reversed_full_coords)-1):
                    reversed_coords = [full_coords[i], full_coords[i + 1]]
                    edge_attr = {'eid': self.eid, 'coords': reversed_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway']}
                    rn.add_edge(reversed_coords[0], reversed_coords[-1], **edge_attr)
                    self.eid += 1


def store_shp(rn, target_path):
    """Store graph as Shapefiles without GDAL/osgeo.

    Output matches the common directory layout:
    - nodes.shp/.shx/.dbf
    - edges.shp/.shx/.dbf
    """
    try:
        shapefile = __import__('shapefile')
    except ImportError as exc:
        raise ImportError(
            "pyshp is required to export Shapefiles. Install it with: pip install pyshp"
        ) from exc

    rn.remove_nodes_from(list(nx.isolates(rn)))
    print('# of nodes:{}'.format(rn.number_of_nodes()))
    print('# of edges:{}'.format(rn.number_of_edges()))

    output_dir = Path(target_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write nodes shapefile.
    nodes_writer = shapefile.Writer(str(output_dir / 'nodes'), shapeType=shapefile.POINT)
    nodes_writer.field('node_id', 'N')
    nodes_writer.field('lon', 'F', decimal=8)
    nodes_writer.field('lat', 'F', decimal=8)

    for node_id, node in enumerate(rn.nodes()):
        lon, lat = node
        nodes_writer.point(lon, lat)
        nodes_writer.record(node_id, lon, lat)
    nodes_writer.close()

    # Write edges shapefile.
    edges_writer = shapefile.Writer(str(output_dir / 'edges'), shapeType=shapefile.POLYLINE)
    edges_writer.field('eid', 'N')
    edges_writer.field('raw_eid', 'N')
    edges_writer.field('highway', 'C', size=40)

    for u, v, data in rn.edges(data=True):
        coords = data.get('coords', [u, v])
        edges_writer.line([coords])
        edges_writer.record(
            int(data.get('eid', -1)),
            int(data.get('raw_eid', -1)),
            str(data.get('highway', ''))
        )
    edges_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='the input path of the original osm data')
    parser.add_argument('--output_path', help='the output directory of the constructed road network')
    opt = parser.parse_args()
    print(opt)

    rn = nx.DiGraph()
    handler = OSM2RNHandler(rn)
    handler.apply_file(opt.input_path, locations=True)
    store_shp(rn, opt.output_path)
