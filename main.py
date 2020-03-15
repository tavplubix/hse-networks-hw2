#!/usr/bin/env python3.6

import sys
import argparse
import os
import math
import csv
import networkx as nx
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xml
from haversine import haversine as geodistance, Unit


class NetGraph:

    def __init__(self, verbose=True):
        self.delay_mks = 4.83
        self.v = verbose
        self.graph = nx.Graph()
        self.nodes = dict()
        self.edges = []
        self.main_routes = {}
        self.reserve_routes = {}
        self.name = None

    def parse_graph(self, file_path):
        self.name = file_path
        if self.v:
            print('Parsing graph from GraphML file:', file_path)
        ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}

        data = xml.parse(file_path).getroot()
        lat_key = data.find("graphml:key[@attr.name='Latitude']", ns).attrib['id']
        lon_key = data.find("graphml:key[@attr.name='Longitude']", ns).attrib['id']
        lab_key = data.find("graphml:key[@attr.name='label']", ns).attrib['id']

        graph_data = data.find("graphml:graph", ns)

        for node in graph_data.findall("graphml:node", ns):
            try:
                node_id = int(node.attrib['id'])
                node_lat = float(node.find(f"graphml:data[@key='{lat_key}']", ns).text)
                node_lon = float(node.find(f"graphml:data[@key='{lon_key}']", ns).text)
                node_label = node.find(f"graphml:data[@key='{lab_key}']", ns).text
                self.nodes[node_id] = {'lat': node_lat, 'lon': node_lon, 'lab': node_label}
            except:
                if self.v:
                    print('Cannot parse node', node, node.attrib, node.text, 'Skipping it')

        for edge in graph_data.findall("graphml:edge", ns):
            try:
                src = int(edge.attrib['source'])
                dst = int(edge.attrib['target'])
                if src not in self.nodes or dst not in self.nodes:
                    if self.v:
                        print(f'Skipping edge {(src, dst)} because some some of this nodes was skipped')
                    continue
                self.edges.append((src, dst))
            except:
                if self.v:
                    print('Cannot parse edge', edge, edge.attrib, edge.text, 'Skipping it')

        if self.v:
            print(f'Parsed {len(self.nodes)} nodes and {len(self.edges)} edges')
        self.setup_graph()

    def setup_graph(self):
        if self.v:
            print('Setting up networkx')
        self.graph.add_nodes_from(self.nodes)
        for src, dst in self.edges:
            src_coord = (self.nodes[src]['lat'], self.nodes[src]['lon'])
            dst_coord = (self.nodes[dst]['lat'], self.nodes[dst]['lon'])
            dist = geodistance(src_coord, dst_coord, Unit.KILOMETERS)
            self.graph.add_edge(src, dst, weight=dist)

        self.edges = list(self.graph.edges(data=True))
        self.edges.sort()
        if self.v:
            print(f'Actually found {self.graph.number_of_nodes()} nodes '
                  f'and {self.graph.number_of_edges()} unique undirected edges')

    def print_edges(self, file_path):
        if self.v:
            print('Writing list of edges to', file_path or 'stdout')
        if file_path is not None:
            out = open(file_path, 'w')
        else:
            out = sys.stdout
        writer = csv.writer(out)

        header = ['Node1 id', 'Node1 label', 'Node1 lat', 'Node1 lon',
                  'Node2 id', 'Node2 label', 'Node2 lat', 'Node2 lon',
                  'Distance (km)', 'Delay (mks)']
        writer.writerow(header)
        for src, dst, param in self.edges:
            src_node = self.nodes[src]
            dst_node = self.nodes[dst]
            dist = param['weight']
            delay = dist * self.delay_mks
            writer.writerow([src, src_node['lab'], src_node['lat'], src_node['lon'],
                             dst, dst_node['lab'], dst_node['lat'], dst_node['lon'],
                             dist, delay])

    def find_main_routes(self):
        if self.v:
            print('Searching for main routes in graph')
        routes, lens = nx.floyd_warshall_predecessor_and_distance(self.graph)
        for src in sorted(self.nodes):
            for dst in sorted(self.nodes):
                if dst <= src or math.isinf(lens[src][dst]):
                    continue
                route = nx.reconstruct_path(src, dst, routes)
                self.main_routes[(src, dst)] = {'route': route.copy(),
                                                'len': lens[src][dst],
                                                'delay': lens[src][dst] * self.delay_mks}
        if self.v:
            print(f'Found {len(self.main_routes)} main routes')

    def try_find_route(self, graph, src, dst):
        try:
            route = nx.dijkstra_path(graph, src, dst)
            length = nx.dijkstra_path_length(graph, src, dst)
            return {'route': route,
                    'len': length,
                    'delay': length * self.delay_mks}
        except nx.NetworkXNoPath:
            return None

    def try_find_reserve_route(self, src, dst):
        if (src, dst) not in self.main_routes:
            return None
        main = self.main_routes[(src, dst)]
        nodes_to_exclude = main['route'][1:-1]

        adjacent = len(nodes_to_exclude) == 0
        if adjacent:
            subgraph = self.graph.copy()
            subgraph.remove_edge(src, dst)
        else:
            nodes = [x for x in self.nodes if x not in nodes_to_exclude]
            subgraph = self.graph.subgraph(nodes)

        route = self.try_find_route(subgraph, src, dst)
        if route is not None:
            return route
        if adjacent:
            return self.main_routes[(src, dst)]

        min_len = math.inf
        for i in range(0, len(nodes_to_exclude)):
            nodes = [x for x in self.nodes if x not in nodes_to_exclude or x == nodes_to_exclude[i]]
            subgraph = self.graph.subgraph(nodes)
            p = self.try_find_route(subgraph, src, dst)
            if p is not None and p['len'] < min_len:
                min_len = p['len']
                route = p.copy()

        return route

    def find_reserve_routes(self):
        if self.v:
            print('Searching for reserve routes in graph')
        for src, dst in self.main_routes:
            rroute = self.try_find_reserve_route(src, dst)
            if rroute is not None:
                self.reserve_routes[(src, dst)] = rroute
        if self.v:
            print(f'Found {len(self.reserve_routes)} reserve routes')

    def print_routes(self, file_path, print_reserve):
        if self.v:
            print('Writing routes to', file_path or 'stdout')
        if file_path is not None:
            out = open(file_path, 'w')
        else:
            out = sys.stdout
        writer = csv.writer(out)
        header = ['Node1', 'Node2', 'Path type', 'Path', 'Delay']

        writer.writerow(header)
        for src, dst in self.main_routes:
            p = self.main_routes[(src, dst)]
            writer.writerow([src, dst, 'main', p['route'], p['delay']])
            if print_reserve and (src, dst) in self.reserve_routes:
                p = self.reserve_routes[(src, dst)]
                writer.writerow([src, dst, 'reserve', p['route'], p['delay']])

    def print_route(self, src, dst, print_reserve):
        print()
        print(f'Route from {src} to {dst}:')
        if src == dst:
            print('Source and destination nodes are the same')
            return
        if dst < src:
            src, dst = dst, src
        print('Node1', 'Node2', 'Path type', 'Path', 'Delay', sep='\t')
        if (src, dst) in self.main_routes:
            p = self.main_routes[(src, dst)]
            print(src, dst, 'main', p['route'], p['delay'], sep='\t')
        if print_reserve and (src, dst) in self.reserve_routes:
            p = self.reserve_routes[(src, dst)]
            print(src, dst, 'main', p['route'], p['delay'], sep='\t')
        print()

    def draw_graph(self, src, dst, file_path):
        labels = dict()
        for node in self.nodes:
            labels[node] = self.nodes[node]['lab']

        try:
            pos = nx.planar_layout(self.graph)
        except nx.NetworkXException:
            if self.v:
                print('Cannot create planar layout')
            pos = nx.spring_layout(self.graph)

        nx.draw_networkx(self.graph, pos=pos, with_labels=False)
        nx.draw_networkx(self.graph, pos=pos, with_labels=False)

        nx.draw_networkx_nodes(self.graph, pos, nodelist=[src, dst], node_color='g')
        if (src, dst) in self.main_routes:
            mr = self.main_routes[(src, dst)]['route']
            edges = []
            for i in range(1, len(mr)):
                edges.append((mr[i-1], mr[i]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='g', width=2)

        if (src, dst) in self.reserve_routes:
            mr = self.reserve_routes[(src, dst)]['route']
            edges = []
            for i in range(1, len(mr)):
                edges.append((mr[i-1], mr[i]))
            nx.draw_networkx_edges(self.graph, pos, edgelist=edges, edge_color='r', width=2)

        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=9)
        plt.title(self.name)
        plt.text(0.5, 0.97, "green route is main, red route is reserve",
                 horizontalalignment='center', transform=plt.gca().transAxes)
        plt.savefig(file_path + '.png')


def main():
    argparser = argparse.ArgumentParser(add_help=True)
    argparser.add_argument('-t', type=str, dest='file_path', help='Path to file with topology in GraphML format', required=True)
    argparser.add_argument('-s', type=int, dest='src', help='source node to print path')
    argparser.add_argument('-d', type=int, dest='dst', help='destination node to print path')
    argparser.add_argument('-r', help='print reserve path', action="store_true")
    argparser.add_argument('-v', help='verbose', action="store_true")
    argparser.add_argument('-i', type=str, dest='img_path', help='path to save .png with route visualization')

    args = argparser.parse_args()
    assert((args.src is None) == (args.dst is None))
    assert(not args.img_path or args.src is not None)

    g = NetGraph()
    g.parse_graph(args.file_path)
    g.print_edges(os.path.basename(args.file_path) + '_topo.csv')
    g.find_main_routes()
    if args.r:
        g.find_reserve_routes()
    g.print_routes(os.path.basename(args.file_path) + '_routes.csv', args.r)
    if args.src:
        g.print_route(args.src, args.dst, args.r)
    if args.img_path:
        g.draw_graph(args.src, args.dst, args.img_path)


if __name__ == "__main__":
    main()
