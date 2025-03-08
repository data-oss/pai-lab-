def n_queen(N):
    def safe(board,row,collum):

        for i in range(collum):
            if board[row][i]==1:
                return False
        for i,j in zip(range(row,-1,-1),range(collum,-1,-1)):
            if board[i][j]==1:
                return False
        for i,j in zip(range(row,N,1),range(collum,-1,-1)):
            if board[i][j]==1:
                    return False
            return True
        
    def solution(board,collum):
        if collum>=N:
            return True
        for i in range(N):
            if safe(board,i,collum):
                board[i][collum]=1
                if solution(board,collum+1):
                    return True
                board[i][collum]=0
        return False
    board = [[0 for _ in range (N)] for _ in range (N)]
    if solution(board,0):
        for row in board:
            print(' '.join(str(x) for x in row))
    else:
        print("Solution not exit ")
N = 8
n_queen(N)


