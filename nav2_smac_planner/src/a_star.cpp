// Copyright (c) 2020, Samsung Research America
// Copyright (c) 2020, Applied Electric Vehicles Pty Ltd
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. Reserved.

#include "nav2_smac_planner/a_star.hpp"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_array.hpp"
using namespace std::chrono;  // NOLINT

namespace nav2_smac_planner
{
template <typename NodeT>
AStarAlgorithm<NodeT>::AStarAlgorithm(
  const MotionModel & motion_model, const SearchInfo & search_info)
: _traverse_unknown(true),
  _max_iterations(0),
  _max_planning_time(0),
  _x_size(0),
  _y_size(0),
  _search_info(search_info),
  _goal_coordinates(Coordinates()),
  _start(nullptr),
  _goal(nullptr),
  _motion_model(motion_model)
{
  _graph.reserve(100000);
}

template <typename NodeT>
AStarAlgorithm<NodeT>::~AStarAlgorithm()
{
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::initialize(
  const bool & allow_unknown, int & max_iterations, const int & max_on_approach_iterations,
  const double & max_planning_time, const float & lookup_table_size,
  const unsigned int & dim_3_size)
{
  _traverse_unknown = allow_unknown;
  _max_iterations = max_iterations;
  _max_on_approach_iterations = max_on_approach_iterations;
  _max_planning_time = max_planning_time;
  NodeT::precomputeDistanceHeuristic(lookup_table_size, _motion_model, dim_3_size, _search_info);
  _dim3_size = dim_3_size;
  _expander = std::make_unique<AnalyticExpansion<NodeT>>(
    _motion_model, _search_info, _traverse_unknown, _dim3_size);
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::setCollisionChecker(GridCollisionChecker * collision_checker)
{
  _collision_checker = collision_checker;
  _costmap = collision_checker->getCostmap();
  unsigned int x_size = _costmap->getSizeInCellsX();
  unsigned int y_size = _costmap->getSizeInCellsY();

  clearGraph();

  if (getSizeX() != x_size || getSizeY() != y_size) {
    _x_size = x_size;
    _y_size = y_size;
    NodeT::initMotionModel(_motion_model, _x_size, _y_size, _dim3_size, _search_info);
  }
  _expander->setCollisionChecker(collision_checker);
}

template <typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr AStarAlgorithm<NodeT>::addToGraph(
  const unsigned int & index)
{
  // Emplace will only create a new object if it doesn't already exist.
  // If an element exists, it will return the existing object, not create a new one.
  return &(_graph.emplace(index, NodeT(index)).first->second);
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::setStart(
  const unsigned int & mx, const unsigned int & my, const unsigned int & dim_3, const double & cmx,
  const double & cmy, const double & cdim_3)
{
  _start = addToGraph(NodeT::getIndex(mx, my, dim_3));
  _start->setPose(
    Coordinates(static_cast<float>(cmx), static_cast<float>(cmy), static_cast<float>(dim_3)));
  _start->setPoseStart(
    Coordinates(static_cast<float>(cmx), static_cast<float>(cmy), static_cast<float>(dim_3)));
  _start->setMotionPrimitiveIndex(0);
  _start->continuous_angle_start = cdim_3;

  // std::cout << "\n\nsetStart: _start->pose_start.x: " << _start->pose_start.x
  //           << ", _start->pose_start.y: " << _start->pose_start.y << std::endl;
  // std::cout << "\n\nsetStart: cmx: " << cmx << ", cmy: " << cmy << std::endl;
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::setGoal(
  const unsigned int & mx, const unsigned int & my, const unsigned int & dim_3, const double & cmx,
  const double & cmy, const double & cdim_3)
{
  // std::cout << "\n\nsetGoal (begin): _start->pose_start.x: " << _start->pose_start.x
  //           << ", _start->pose_start.y: " << _start->pose_start.y << std::endl;

  _goal = addToGraph(NodeT::getIndex(mx, my, dim_3));

  typename NodeT::Coordinates goal_coords(
    static_cast<float>(cmx), static_cast<float>(cmy), static_cast<float>(dim_3));

  if (!_search_info.cache_obstacle_heuristic || goal_coords != _goal_coordinates) {
    if (!_start) {
      throw std::runtime_error("Start must be set before goal.");
    }

    NodeT::resetObstacleHeuristic(_costmap, _start->pose_start.x, _start->pose_start.y, mx, my);
  }

  _goal_coordinates = goal_coords;
  _goal->setPose(_goal_coordinates);
  _goal->is_goal = true;
  _goal->continuous_angle_goal = cdim_3;

  // std::cout << "\n\nsetGoal: _goal->pose.x: " << _goal->pose.x
  //           << ", _goal->pose.y: " << _goal->pose.y << std::endl;
  // std::cout << "\n\nsetGoal: cmx: " << cmx << ", cmy: " << cmy << std::endl;

  // std::cout << "\n\nsetGoal (end): _start->pose_start.x: " << _start->pose_start.x
  //           << ", _start->pose_start.y: " << _start->pose_start.y << std::endl;
}

template <typename NodeT>
bool AStarAlgorithm<NodeT>::areInputsValid()
{
  // Check if graph was filled in
  if (_graph.empty()) {
    throw std::runtime_error("Failed to compute path, no costmap given.");
  }

  // Check if points were filled in
  if (!_start || !_goal) {
    throw std::runtime_error("Failed to compute path, no valid start or goal given.");
  }

  // Check if ending point is valid
  if (
    /*getToleranceHeuristic() < 0.001 && */ !_goal->isNodeValid(
      _traverse_unknown, _collision_checker)) {
    throw std::runtime_error("Failed to compute path, goal is occupied with no tolerance.");
  }

  // Check if starting point is valid
  if (!_start->isNodeValid(_traverse_unknown, _collision_checker)) {
    throw std::runtime_error("Starting point in lethal space! Cannot create feasible plan.");
  }

  return true;
}

template <typename NodeT>
bool AStarAlgorithm<NodeT>::createPath(
  CoordinateVector & path, int & iterations, const float & tolerance)
{
#ifdef PUB_EXPANSION
  static auto node = std::make_shared<rclcpp::Node>("test");
  static auto pub = node->create_publisher<geometry_msgs::msg::PoseArray>("expansions", 1);
  geometry_msgs::msg::PoseArray msg;
  geometry_msgs::msg::Pose msg_pose;
  msg.header.stamp = node->now();
  msg.header.frame_id = "map";
#endif

  steady_clock::time_point start_time = steady_clock::now();
  _tolerance = tolerance;
  _best_heuristic_node = {std::numeric_limits<float>::max(), 0};
  clearQueue();

  if (!areInputsValid()) {
#ifdef PUB_EXPANSION
    pub->publish(msg);
#endif
    return false;
  }

  // 0) Add starting point to the open set
  addNode(0.0, getStart());
  getStart()->setAccumulatedCost(0.0);

  // Optimization: preallocate all variables
  NodePtr current_node = nullptr;
  NodePtr neighbor = nullptr;
  NodePtr expansion_result = nullptr;
  float g_cost = 0.0;
  NodeVector neighbors;
  int approach_iterations = 0;
  NeighborIterator neighbor_iterator;
  int analytic_iterations = 0;
  int closest_distance = std::numeric_limits<int>::max();

  // Given an index, return a node ptr reference if its collision-free and valid
  const unsigned int max_index = getSizeX() * getSizeY() * getSizeDim3();
  NodeGetter neighborGetter = [&, this](
                                const unsigned int & index, NodePtr & neighbor_rtn) -> bool {
    if (index < 0 || index >= max_index) {
      return false;
    }

    // if (this->_goal->getIndex() == index) {
    //   std::cout << "Goal is neighbor" << std::endl;
    // }

    neighbor_rtn = addToGraph(index);
    return true;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////
  // Special case checker that looks if the start and goal pose are exactly in front of eachother.
  // This happens a lot when doing lane coverage at Relox.

  auto start = getStart();
  // https://stackoverflow.com/questions/1878907/how-can-i-find-the-difference-between-two-angles
  float angle_diff = start->continuous_angle_start - _goal->continuous_angle_goal;
  angle_diff = fabs(atan2(sin(angle_diff), cos(angle_diff)));
  // std::cout << "\n\nANGLE DIFF IS: " << angle_diff << "\n\n" << std::endl;
  if (angle_diff < 15.0 * M_PI / 180.0) {
    float dx = _goal->pose.x - start->pose_start.x;
    float dy = _goal->pose.y - start->pose_start.y;
    // std::cout << "\n\nSTART POSE X: " << _start->pose_start.x
    //           << ", START POSE Y: " << _start->pose_start.y << "\n\n"
    //           << std::endl;
    // std::cout << "\n\nGOAL POSE X: " << _goal->pose.x << ", GOAL POSE Y: " << _goal->pose.y
    //           << "\n\n"
    //           << std::endl;
    float l = sqrt(dx * dx + dy * dy);
    // dir of theta rotated 90 degrees counter clockwise (left)
    float left_dir_x = -sin(start->continuous_angle_start);
    float left_dir_y = cos(start->continuous_angle_start);
    float left_dot = (dx / l) * left_dir_x + (dy / l) * left_dir_y;
    // dir of theta (forward)
    float forward_dir_x = cos(start->continuous_angle_start);
    float forward_dir_y = sin(start->continuous_angle_start);
    float forward_dot = (dx / l) * forward_dir_x + (dy / l) * forward_dir_y;
    // std::cout << "\n\n*********\n\nGOAL AND START THETA ARE THE SAME!!! fabs(left_dot)-> ("
    //           << fabs(left_dot) << ") forward_dot -> (" << forward_dot << ") angle_diff -> ( "
    //           << angle_diff << " ) l -> ( " << l << " )\n\n*********\n\n"
    //           << std::endl;

    if (fabs(left_dot) < 0.2 && forward_dot >= 0.0) {
      // std::cout << "\n\n********\n\nAND THEY ARE ALSO COLLINEAR!!!\n\n********\n\n" << std::endl;

      // A move of sqrt(2) is guaranteed to be in a new cell
      static const float sqrt_2 = std::sqrt(2.);
      unsigned int num_intervals = std::floor(l / sqrt_2);

      bool ok = true;
      // its ok to traverse over cost if we are closer than 3 meters
      const float max_l = 3.0 / _costmap->getResolution();

      // Check intermediary poses (non-goal, non-start)
      for (float i = 1; i < num_intervals; i++) {
        float ix = start->pose_start.x + dx * (i / num_intervals);
        float iy = start->pose_start.y + dy * (i / num_intervals);
        // std::cout << "start x: " << start->pose_start.x << " ix: " << ix << " goal x"
        //           << _goal->pose.x << std::endl;
        if (_collision_checker->inCollision(ix, iy, start->pose_start.theta, false)) {
          // not ok!
          // std::cout << "IN COLLISION -> NOT OK!!!" << std::endl;
          ok = false;
          break;
        } else {
          // std::cout << "NOT IN COLLISION!!!" << std::endl;
          if (_collision_checker->getCost() == 0.0 || l < max_l) {
            // ok!
            // std::cout << "COST IS ZERO OR l < max_l -> OK!!! l=" << l << std::endl;
          } else {
            // not ok!
            // std::cout << "COST IS HIGH!!! -> NOT OK l=" << l << std::endl;
            ok = false;
            break;
          }
        }
      }

      if (ok) {
        _goal->parent = start;
#ifdef PUB_EXPANSION
        pub->publish(msg);
#endif
        return _goal->backtracePath(path);
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////

  while (iterations < getMaxIterations() && !_queue.empty()) {
    // Check for planning timeout only on every Nth iteration
    if (iterations % _timing_interval == 0) {
      std::chrono::duration<double> planning_duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(steady_clock::now() - start_time);
      if (static_cast<double>(planning_duration.count()) >= _max_planning_time) {
#ifdef PUB_EXPANSION
        pub->publish(msg);
#endif
        return false;
      }
    }

    // 1) Pick Nbest from O s.t. min(f(Nbest)), remove from queue
    current_node = getNextNode();

    // We allow for nodes to be queued multiple times in case
    // shorter paths result in it, but we can visit only once
    if (current_node->wasVisited()) {
      continue;
    }

#ifdef PUB_EXPANSION
    msg_pose.position.x =
      _costmap->getOriginX() + (current_node->pose.x * _costmap->getResolution());
    msg_pose.position.y =
      _costmap->getOriginY() + (current_node->pose.y * _costmap->getResolution());
    tf2::Quaternion q;
    q.setEuler(0.0, 0.0, NodeHybrid::motion_table.getAngleFromBin(current_node->pose.theta));
    msg_pose.orientation = tf2::toMsg(q);
    msg.poses.push_back(msg_pose);
#endif

    iterations++;

    // 2) Mark Nbest as visited
    current_node->visited();

    // 2.1) Use an analytic expansion (if available) to generate a path
    expansion_result = nullptr;
    expansion_result = _expander->tryAnalyticExpansion(
      current_node, getGoal(), neighborGetter, analytic_iterations, closest_distance);
    if (expansion_result != nullptr) {
      // std::cout << "\n\nUSED ANALYTICAL EXPANSION!!!!!!!!!!!!!\n\n" << std::endl;
      current_node = expansion_result;
    }

    // 3) Check if we're at the goal, backtrace if required
    // std::cout << _best_heuristic_node.first << " < " << getToleranceHeuristic() << " | "
    //           << approach_iterations << " >= " << getOnApproachMaxIterations() << std::endl;
    if (isGoal(current_node)) {
      _goal->parent = current_node->parent;
      _goal->visited();
// std::cout << "_goal -> x: " << _goal->pose.x << " y: " << _goal->pose.y
//           << " theta: " << _goal->pose.theta << std::endl;
#ifdef PUB_EXPANSION
      pub->publish(msg);
#endif
      return _goal->backtracePath(path);
    } else if (_best_heuristic_node.first < getToleranceHeuristic()) {
      // Optimization: Let us find when in tolerance and refine within reason
      approach_iterations++;
      if (approach_iterations >= getOnApproachMaxIterations()) {
        // std::cout << "\n\nCOULD ONLY REACHED GOAL WITHIN TOLERANCE\n\n" << std::endl;
        _goal->parent = _graph.at(_best_heuristic_node.second).parent;
        _goal->visited();
#ifdef PUB_EXPANSION
        pub->publish(msg);
#endif
        return _goal->backtracePath(path);
      }
    }

    // 4) Expand neighbors of Nbest not visited
    neighbors.clear();
    current_node->getNeighbors(neighborGetter, _collision_checker, _traverse_unknown, neighbors);

    for (neighbor_iterator = neighbors.begin(); neighbor_iterator != neighbors.end();
         ++neighbor_iterator) {
      neighbor = *neighbor_iterator;

      // 4.1) Compute the cost to go to this node
      g_cost = current_node->getAccumulatedCost() + current_node->getTraversalCost(neighbor);

      // 4.2) If this is a lower cost than prior, we set this as the new cost and new approach
      if (g_cost < neighbor->getAccumulatedCost()) {
        neighbor->setAccumulatedCost(g_cost);
        neighbor->parent = current_node;

        // 4.3) Add to queue with heuristic cost
        addNode(g_cost + getHeuristicCost(neighbor), neighbor);
      }
    }
  }

#ifdef PUB_EXPANSION
  pub->publish(msg);
#endif
  return false;
}

template <typename NodeT>
bool AStarAlgorithm<NodeT>::isGoal(NodePtr & node)
{
  return node == getGoal();
}

template <typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr & AStarAlgorithm<NodeT>::getStart()
{
  return _start;
}

template <typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr & AStarAlgorithm<NodeT>::getGoal()
{
  return _goal;
}

template <typename NodeT>
typename AStarAlgorithm<NodeT>::NodePtr AStarAlgorithm<NodeT>::getNextNode()
{
  NodeBasic<NodeT> node = _queue.top().second;
  _queue.pop();
  node.processSearchNode();
  return node.graph_node_ptr;
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::addNode(const float & cost, NodePtr & node)
{
  NodeBasic<NodeT> queued_node(node->getIndex());
  queued_node.populateSearchNode(node);
  _queue.emplace(cost, queued_node);
}

template <typename NodeT>
float AStarAlgorithm<NodeT>::getHeuristicCost(const NodePtr & node)
{
  const Coordinates node_coords = NodeT::getCoords(node->getIndex(), getSizeX(), getSizeDim3());
  float heuristic = NodeT::getHeuristicCost(node_coords, _goal_coordinates, _costmap);

  if (heuristic < _best_heuristic_node.first) {
    _best_heuristic_node = {heuristic, node->getIndex()};
  }

  return heuristic;
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::clearQueue()
{
  NodeQueue q;
  std::swap(_queue, q);
}

template <typename NodeT>
void AStarAlgorithm<NodeT>::clearGraph()
{
  Graph g;
  std::swap(_graph, g);
  _graph.reserve(100000);
}

template <typename NodeT>
int & AStarAlgorithm<NodeT>::getMaxIterations()
{
  return _max_iterations;
}

template <typename NodeT>
int & AStarAlgorithm<NodeT>::getOnApproachMaxIterations()
{
  return _max_on_approach_iterations;
}

template <typename NodeT>
float & AStarAlgorithm<NodeT>::getToleranceHeuristic()
{
  return _tolerance;
}

template <typename NodeT>
unsigned int & AStarAlgorithm<NodeT>::getSizeX()
{
  return _x_size;
}

template <typename NodeT>
unsigned int & AStarAlgorithm<NodeT>::getSizeY()
{
  return _y_size;
}

template <typename NodeT>
unsigned int & AStarAlgorithm<NodeT>::getSizeDim3()
{
  return _dim3_size;
}

// Instantiate algorithm for the supported template types
template class AStarAlgorithm<NodeHybrid>;

}  // namespace nav2_smac_planner
