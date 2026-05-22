#具体怎么做：

  #第 1 步：先把画面看懂
  #屏幕上的推箱子游戏是网格的。你把画面切成格子（比如 30×30
  #像素一格），每个格子里要么是空地、要么是墙、要么是箱子、要么是目标。你已经在做颜色识别了，用同样的方法把地图画出来。

  #第 2 步：A 寻路（20 行代码）*
  # 给你一个能直接用的 A* 寻路
  import heapq

  def astar(grid, start, goal):
      # grid[y][x] = 0(空地) or 1(墙)
      # start/goal = (x, y)
      h, w = len(grid[0]), len(grid)
      open_set = [(0, start)]
      came_from = {}
      g_score = {start: 0}

      while open_set:
          _, current = heapq.heappop(open_set)
          if current == goal:
              path = [current]
              while current in came_from:
                  current = came_from[current]
                  path.append(current)
              return path[::-1]  # 反转，起点到终点

          for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
              nx, ny = current[0]+dx, current[1]+dy
              if 0 <= nx < w and 0 <= ny < h and grid[ny][nx] == 0:
                  new_g = g_score[current] + 1
                  if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                      g_score[(nx, ny)] = new_g
                      f = new_g + abs(nx-goal[0]) + abs(ny-goal[1])
                      heapq.heappush(open_set, (f, (nx, ny)))
                      came_from[(nx, ny)] = current
      return None  # 无路

#  第 3 步：推箱子策略
 # 推箱子就三步：
  #1. 找个没在目标上的箱子
  #2. A* 寻路到箱子背后（箱子背对目标的那一侧）
  #3. 把车往箱子方向推（发 $V 指令让你朝箱子推过去）
