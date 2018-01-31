/*
 ==============================================================================

 RPGraph.cpp
 Copyright © 2016, 2017, 2018  G. Brinkmann

 This file is part of graph_viewer.

 graph_viewer is free software: you can redistribute it and/or modify
 it under the terms of version 3 of the GNU Affero General Public License as
 published by the Free Software Foundation.

 graph_viewer is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with graph_viewer.  If not, see <https://www.gnu.org/licenses/>.

 ==============================================================================
*/


#include <stdio.h>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <algorithm>
#include "RPGraph.hpp"

namespace RPGraph
{
    /* Definitions for UGraph */
    UGraph::UGraph(std::string edgelist_path)
    {
        node_count = 0;
        edge_count = 0;

        std::fstream edgelist_file(edgelist_path, std::ifstream::in);

        std::string line;
        while(std::getline(edgelist_file, line))
        {
            // Skip any comments
            if(line[0] == '#') continue;

            // Read source and target from file
            nid_t s, t;
            std::istringstream(line) >> s >> t;

            if(s != t and !has_edge(s, t)) add_edge(s, t);
        }

        edgelist_file.close();
    }

    bool UGraph::has_node(nid_t nid)
    {
        return node_map.count(nid) > 0;
    }

    bool UGraph::has_edge(nid_t s, nid_t t)
    {
        if(!has_node(s) or !has_node(t)) return false;

        nid_t s_mapped = node_map[s];
        nid_t t_mapped = node_map[t];

        if(adjacency_list.count(std::min(s_mapped, t_mapped)) == 0) return false;

        std::vector<nid_t> neighbors = adjacency_list[std::min(s_mapped, t_mapped)];
        if(std::find(neighbors.begin(), neighbors.end(), std::max(s_mapped, t_mapped)) == neighbors.end())
            return false;
        else
            return true;
    }

    void UGraph::add_node(nid_t nid)
    {
        if(!has_node(nid))
        {
            node_map[nid] = node_count;
            node_map_r[node_count] = nid;
            node_count++;
        }
    }

    void UGraph::add_edge(nid_t s, nid_t t)
    {
        if(has_edge(s, t)) return;
        if(!has_node(s)) add_node(s);
        if(!has_node(t)) add_node(t);
        nid_t s_mapped = node_map[s];
        nid_t t_mapped = node_map[t];

        // Insert edge into adjacency_list
        adjacency_list[std::min(s_mapped, t_mapped)].push_back(std::max(s_mapped, t_mapped));
        degrees[s_mapped] += 1;
        degrees[t_mapped] += 1;
        edge_count++;
    }

    nid_t UGraph::num_nodes()
    {
        return node_count;
    }

    nid_t UGraph::num_edges()
    {
        return edge_count;
    }

    nid_t UGraph::degree(nid_t nid)
    {
        return degrees[nid];
    }

    nid_t UGraph::in_degree(nid_t nid)
    {
        return degree(nid);
    }

    nid_t UGraph::out_degree(nid_t nid)
    {
        return degree(nid);
    }
    std::vector<nid_t> UGraph::neighbors_with_geq_id(nid_t nid)
    {
        return adjacency_list[nid];
    }

    /* Definitions for CSRUGraph */

// CSRUGraph represents an undirected graph using a
// compressed sparse row (CSR) datastructure.
    CSRUGraph::CSRUGraph(nid_t num_nodes, nid_t num_edges)
    {
        // `edges' is a concatenation of all edgelists
        // `offsets' contains offset (in `edges`) for each nodes' edgelist.
        // `nid_to_offset` maps nid to index to be used in `offset'

        // e.g. the edgelist of node with id `nid' starts at
        // edges[offsets[nid_to_offset[nid]]] and ends at edges[offsets[nid_to_offset[nid]] + 1]
        // (left bound inclusive right bound exclusive)

        edge_count = num_edges; // num_edges counts each bi-directional edge once.
        node_count = num_nodes;
        edges =   (nid_t *) malloc(sizeof(nid_t) * 2 * edge_count);
        offsets = (nid_t *) malloc(sizeof(nid_t) * node_count);
        offset_to_nid = (nid_t *) malloc(sizeof(nid_t) * node_count);

        // Create a map from original ids to ids used throughout CSRUGraph
        nid_to_offset.reserve(node_count);

        first_free_id = 0;
        edges_seen = 0;
    }

    CSRUGraph::~CSRUGraph()
    {
        free(edges);
        free(offsets);
        free(offset_to_nid);
    }

    void CSRUGraph::insert_node(nid_t node_id, std::vector<nid_t> nbr_ids)
    {
        nid_t source_id_old = node_id;
        nid_t source_id_new = first_free_id;
        nid_to_offset[source_id_old] = first_free_id;
        offset_to_nid[first_free_id] = source_id_old;
        first_free_id++;

        offsets[source_id_new] = edges_seen;
        for (auto nbr_id : nbr_ids)
        {
            nid_t dest_id_old = nbr_id;
            edges[edges_seen] = dest_id_old;
            edges_seen++;
        }
    }

    void CSRUGraph::fix_edge_ids()
    {
        for (eid_t ei = 0; ei < 2*edge_count; ei++)
        {
            edges[ei] = nid_to_offset[edges[ei]];
        }
    }

    nid_t CSRUGraph::degree(nid_t nid)
    {
        // If nid is last element of `offsets'... we prevent out of bounds.
        nid_t r_bound;
        if (nid < node_count - 1) r_bound = offsets[nid+1];
        else r_bound = edge_count * 2;
        nid_t l_bound = offsets[nid];
        return (r_bound - l_bound);
    }

    nid_t CSRUGraph::out_degree(nid_t nid)
    {
        return degree(nid);
    }

    nid_t CSRUGraph::in_degree(nid_t nid)
    {
        return degree(nid);
    }

    nid_t CSRUGraph::nbr_id_for_node(nid_t nid, nid_t edge_no)
    {
        return edges[offsets[nid] + edge_no];
    }
    nid_t CSRUGraph::num_nodes()
    {
        return node_count;
    }

    nid_t CSRUGraph::num_edges()
    {
        return edge_count;
    }
};
