public class Solution {
    private static final int MOD = 1000000007;

    public int solution(int[] A, int C1, int C2) {
        int n = A.length;
        int maxz = Arrays.stream(A).max().getAsInt();
        long res = Long.MAX_VALUE;
        
        for (int i = 0; i <= n; i++) {
            PriorityQueue<Integer> pq = new PriorityQueue<>();
            for (int num : A) {
                pq.offer(num);
            }
            
            int cnt_c1 = 0, cnt_c2 = 0;
            
            while (!pq.isEmpty()) {
                int min_num = pq.poll();
                int sec_min = pq.poll();
                
                if (sec_min == maxz + i) {
                    cnt_c1 = maxz + i - min_num;
                    break;
                }
                
                min_num++;
                sec_min++;
                cnt_c2++;
                pq.offer(min_num);
                pq.offer(sec_min);
            }
            
            res = Math.min(res, (cnt_c1 * C1 + cnt_c2 * C2));
        }
        
        return (int) (res % MOD);
    }
}