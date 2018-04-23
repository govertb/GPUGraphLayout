/*
 ==============================================================================

 RPGraphLayout.cpp
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


#include "RPGraphLayout.hpp"
#include "../lib/pngwriter/src/pngwriter.h"

#include <fstream>
#include <cmath>
#include <limits>

namespace RPGraph
{
    GraphLayout::GraphLayout(UGraph &graph, float width, float height)
        : graph(graph), width(width), height(height)
    {
        coordinates = (Coordinate *) malloc(graph.num_nodes() * sizeof(Coordinate));
        minx = std::numeric_limits<float>::max();
        miny = std::numeric_limits<float>::max();
        maxx = std::numeric_limits<float>::min();
        maxy = std::numeric_limits<float>::min();
    }

    GraphLayout::~GraphLayout()
    {
        free(coordinates);
    }

    void GraphLayout::randomizePositions()
    {
        for (nid_t i = 0; i <  graph.num_nodes(); ++i)
        {
            setX(i, get_random(-width/2.0, width/2.0));
            setY(i, get_random(-height/2.0, height/2.0));
        }
    }

    float GraphLayout::getX(nid_t node_id)
    {
        return coordinates[node_id].x;
    }

    float GraphLayout::getY(nid_t node_id)
    {
        return coordinates[node_id].y;
    }

    float GraphLayout::minX()
    {
        return this->minx;
    }

    float GraphLayout::maxX()
    {
        return this->maxx;
    }

    float GraphLayout::minY()
    {
        return this->miny;
    }

    float GraphLayout::maxY()
    {
        return this->maxy;
    }

    float GraphLayout::getXRange()
    {
        return maxX()- minX();
    }

    float GraphLayout::getYRange()
    {
        return maxY() - minY();
    }

    float GraphLayout::getSpan()
    {
        return fmaxf(getXRange(), getYRange());
    }

    float GraphLayout::getDistance(nid_t n1, nid_t n2)
    {
        const float dx = getX(n1)-getX(n2);
        const float dy = getY(n1)-getY(n2);
        return std::sqrt(dx*dx + dy*dy);
    }

    Real2DVector GraphLayout::getDistanceVector(nid_t n1, nid_t n2)
    {
        return Real2DVector(getX(n2) - getX(n1), getY(n2) - getY(n1));
    }

    Real2DVector GraphLayout::getNormalizedDistanceVector(nid_t n1, nid_t n2)
    {
        const float x1 = getX(n1);
        const float x2 = getX(n2);
        const float y1 = getY(n1);
        const float y2 = getY(n2);
        const float dx = x2 - x1;
        const float dy = y2 - y1;
        const float len = std::sqrt(dx*dx + dy*dy);

        return Real2DVector(dx / len, dy / len);
    }

    Coordinate GraphLayout::getCoordinate(nid_t node_id)
    {
        return coordinates[node_id];
    }

    Coordinate GraphLayout::getCenter()
    {
        float x = minX() + getXRange()/2.0;
        float y = minY() + getYRange()/2.0;
        return Coordinate(x, y);
    }

    void GraphLayout::setX(nid_t node_id, float x_value)
    {
        minx = fminf(minx, x_value);
        maxx = fmaxf(maxx, x_value);
        coordinates[node_id].x = x_value;
    }

    void GraphLayout::setY(nid_t node_id, float y_value)
    {
        miny = fminf(miny, y_value);
        maxy = fmaxf(maxy, y_value);
        coordinates[node_id].y = y_value;
    }

    void GraphLayout::moveNode(nid_t n, RPGraph::Real2DVector v)
    {
        setX(n, getX(n) + v.x);
        setY(n, getY(n) + v.y);
    }

    void GraphLayout::setCoordinates(nid_t node_id, Coordinate c)
    {
        setX(node_id, c.x);
        setY(node_id, c.y);
    }

    void GraphLayout::writeToPNG(const int width, const int height, const char *path)
    {
        const float xRange = getXRange();
        const float yRange = getYRange();
        const RPGraph::Coordinate center = getCenter();
        const float xCenter = center.x;
        const float yCenter = center.y;
        const float minX = xCenter - xRange/2.0;
        const float minY = yCenter - yRange/2.0;
        const float xScale = width/xRange;
        const float yScale = height/yRange;

        // Here we need to do some guessing as to what the optimal
        // opacity of nodes and edges might be, given how many of them we need to draw.
        const float node_opacity = 1/(0.0001  * graph.num_nodes());
        const float edge_opacity = 1/(0.00001 * graph.num_edges());

        // Write to file.
        pngwriter layout_png(width, height, 0, path);
        layout_png.invert(); // set bg. to white

        for (nid_t n1 = 0; n1 < graph.num_nodes(); ++n1)
        {
            // Plot node,
            layout_png.filledcircle_blend((getX(n1) - minX)*xScale,
                                          (getY(n1) - minY)*yScale,
                                          3, node_opacity, 0, 0, 0);
            for (nid_t n2 : graph.neighbors_with_geq_id(n1)) {
                // ... and edge.
                layout_png.line_blend((getX(n1) - minX)*xScale, (getY(n1) - minY)*yScale,
                                      (getX(n2) - minX)*xScale, (getY(n2) - minY)*yScale,
                                      edge_opacity, 0, 0, 0);
            }
        }
        // Write it to disk.
        layout_png.write_png();
    }

    void GraphLayout::writeToCSV(const char *path)
    {
        if (is_file_exists(path))
        {
            printf("Error: File exists at %s\n", path);
            exit(EXIT_FAILURE);
        }

        std::ofstream out_file(path);

        for (nid_t n = 0; n < graph.num_nodes(); ++n)
        {
            nid_t id = graph.node_map_r[n]; // id as found in edgelist
            out_file << id << "," << getX(n) << "," << getY(n) << "\n";
        }

        out_file.close();
    }

    void GraphLayout::writeToBin(const char *path)
    {
        if (is_file_exists(path))
        {
            printf("Error: File exists at %s\n", path);
            exit(EXIT_FAILURE);
        }

        std::ofstream out_file(path, std::ofstream::binary);

        for (nid_t n = 0; n < graph.num_nodes(); ++n)
        {
            nid_t id = graph.node_map_r[n]; // id as found in edgelist
            float x = getX(n);
            float y = getY(n);

            out_file.write(reinterpret_cast<const char*>(&id), sizeof(id));
            out_file.write(reinterpret_cast<const char*>(&x), sizeof(x));
            out_file.write(reinterpret_cast<const char*>(&y), sizeof(y));
        }

        out_file.close();
    }

}
