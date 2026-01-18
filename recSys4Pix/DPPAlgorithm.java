

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Profile;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * @author OysterQAQ
 * @version 1.0
 * @date 2023/5/25 15:36
 * @description Dpp
 */

@Component
@Slf4j
public class DPPAlgorithm {
    private static final int threadNum = 48;

    private static final int corePoolSize = 48;

    private static final int maximumPoolSize = 48;

    private static final int keepAliveTime = 1;

    private static final int capacity = 1000000;
    private static final float epsilon = 1e-20F;

    private static ExecutorService threadPool = new ThreadPoolExecutor(corePoolSize, maximumPoolSize, keepAliveTime,
            TimeUnit.HOURS, new LinkedBlockingDeque<>(capacity));

    public List<Integer> reArrangeByDPP(List<Integer> items, float[] scores, float[][] featureMatrix, int maxLength) {
        try {
            //构造相似度分数矩阵：特征矩阵乘以自身的转置，即得到特征矩阵中特征向量相互之间的点乘，用以表示距离
            float[][] similarityMatrix = calculateSimilarityMatrix(featureMatrix);
            //构造核矩阵：相似矩阵在分别乘上两个特征向量的分数
            float[][] kernelMatrix = calculateKernelMatrix(scores, similarityMatrix);
            // 调用DPP算法（核矩阵再进入DPP）
            return DPP(kernelMatrix, items, maxLength);
        } catch (InterruptedException e) {
            log.error("error happened.", e);
            return items;
        }

    }

    private List<Integer> DPP(float[][] kernelMatrix, List<Integer> items, int maxLength) {
        //获取项目数量
        int itemSize = kernelMatrix.length;
        //存储中间值的二维数组
        float[][] cis = new float[maxLength][itemSize];
        //最终结果
        List<Integer> selectedItemList = new ArrayList<>(maxLength);
        //取kernelMatrix对角线元素
        float[] di2s = fetchDiagonal(kernelMatrix);
        //找到其中最大值的index
        int selectedItemIndex = findMaxIndex(di2s);
        //根据index找到所选取物品id 加入列表
        selectedItemList.add(items.get(selectedItemIndex));
        while (selectedItemList.size() < maxLength) {
            //当前已选择的项目数量
            final int k = selectedItemList.size() - 1;
            //取cis[0,k)行，第selectedItemIndex列 作为ci
            final float[] ciOptimal = fetchCiOptimal(0, k, selectedItemIndex, cis);
            //di2s[selectedItemIndex]开方
            final float diOptimal = (float) Math.sqrt(di2s[selectedItemIndex]);
            //取当前选择项目和其余项目的相似度向量
            final float[] elements = fetchElements(selectedItemIndex, kernelMatrix);
            //计算eis
            final float[] eis = calculateEis(k, ciOptimal, cis, elements, diOptimal, 0);
            //将eis赋值给 cis 的第 k 行
            System.arraycopy(eis, 0, cis[k], 0, cis[k].length);
            //更新di2s 逐元素和eis对应元素的平方相减
            di2sMinusSquareOfEis(di2s, eis);
            //将已选择项目的di2s设为负无穷，以排除再次选择
            di2s[selectedItemIndex] = Float.MIN_VALUE;
            //找到最大值，不将这个过程与上一步一起进行的原因是期待jvm将上一步不带分支的float数组运算进行向量优化
            selectedItemIndex = findMaxIndex(di2s);
            if (di2s[selectedItemIndex] < epsilon) {
                break;
            }
            selectedItemList.add(items.get(selectedItemIndex));

        }
        return selectedItemList;

    }

    private void di2sMinusSquareOfEis(float[] di2s, float[] eis) {
        for (int i = 0; i < di2s.length; i++) {
            di2s[i] -= eis[i] * eis[i];
        }
    }

    public List<Integer> DPPSW(float[][] kernelMatrix, List<Integer> items, int maxLength, int windowSize) {
        int itemSize = kernelMatrix.length;
        float[][] v = new float[maxLength][maxLength];
        float[][] cis = new float[maxLength][itemSize];
        List<Integer> selectedItemList = new ArrayList<>(maxLength);
        float[] di2s = fetchDiagonal(kernelMatrix);
        int selectedItemIndex = findMaxIndex(di2s);
        selectedItemList.add(items.get(selectedItemIndex));
        int windowLeftIndex = 0;

        while (selectedItemList.size() < maxLength) {
            int k = selectedItemList.size() - 1;
            float[] ciOptimal = fetchCiOptimal(windowLeftIndex, k, selectedItemIndex, cis);
            float diOptimal = (float) Math.sqrt(di2s[selectedItemIndex]);
            updateVByCIOptimal(v, ciOptimal, k, windowLeftIndex, k);
            v[k][k] = diOptimal;
            float[] elements = fetchElements(selectedItemIndex, kernelMatrix);
            float[] eis = calculateEis(k, ciOptimal, cis, elements, diOptimal, windowLeftIndex);
            System.arraycopy(eis, 0, cis[k], 0, cis[k].length);
            di2sMinusSquareOfEis(di2s, eis);
            if (selectedItemList.size() >= windowSize) {
                windowLeftIndex++;
                for (int ind = windowLeftIndex; ind <= k; ind++) {
                    float t = (float) Math.sqrt(Math.pow(v[ind][ind], 2) + Math.pow(v[ind][windowLeftIndex - 1], 2));
                    float c = t / v[ind][ind];
                    float s = v[ind][windowLeftIndex - 1] / v[ind][ind];
                    v[ind][ind] = t;
                    for (int i = ind + 1; i <= k; i++) {
                        v[i][ind] += s * v[i][windowLeftIndex - 1];
                        v[i][ind] /= c;
                        v[i][windowLeftIndex - 1] *= c;
                        v[i][windowLeftIndex - 1] -= s * v[i][ind];
                    }

                    for (int i = 0; i < cis[ind].length; i++) {
                        cis[ind][i] += s * cis[windowLeftIndex - 1][i];
                        cis[ind][i] /= c;
                        cis[windowLeftIndex - 1][i] *= c;
                        cis[windowLeftIndex - 1][i] -= s * cis[ind][i];
                    }
                }
                disAddSquareOfCi(di2s, cis[windowLeftIndex - 1]);
            }
            di2s[selectedItemIndex] = Float.MIN_VALUE;
            selectedItemIndex = findMaxIndex(di2s);
            if (di2s[selectedItemIndex] < epsilon) {
                break;
            }
            selectedItemList.add(items.get(selectedItemIndex));
        }
        return selectedItemList;
    }

    private void disAddSquareOfCi(float[] di2s, float[] ci) {
        for (int i = 0; i < ci.length; i++) {
            di2s[i] += ci[i] * ci[i];
        }
    }

    private void updateVByCIOptimal(float[][] v, float[] ciOptimal, int rowIndex, int updateStartIndex, int updateEndIndex) {
        for (int i = updateStartIndex; i < updateEndIndex; i++) {
            v[rowIndex][i] = ciOptimal[i - updateStartIndex];
        }
    }

    private int findMaxIndex(float[] di2s) {
        float max = di2s[0];
        int maxIndex = 0;
        for (int i = 1; i < di2s.length; i++) {
            if (di2s[i] > max) {
                max = di2s[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private int updateDi2sByEisAndFindMaxIndex(float[] di2s, float[] eis) {
        float maxValue = Float.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < di2s.length; i++) {
            di2s[i] -= eis[i] * eis[i];
            if (di2s[i] > maxValue) {
                maxValue = di2s[i];
                index = i;
            }
        }
        return index;
    }

    private float[] calculateEis(int rowEndIndex, float[] ciOptimal, float[][] cis, float[] elements, float diOptimal,
                                 int rowStartIndex) {
        float[] eis = new float[elements.length];
        //选择cis前k行 逐行和ciOptimal求点积
        //elements-得到的结果再除以diOptimal
        for (int i = 0; i < eis.length; i++) {
            float rt = 0;
            for (int j = rowStartIndex; j < rowEndIndex; j++) {
                rt += ciOptimal[j] * cis[j][i];
            }
            eis[i] = (elements[i] - rt) / diOptimal;
        }
        return eis;
    }

    private float[] fetchElements(int selectedItemIndex, float[][] kernelMatrix) {
        //取第selectedItemIndex行的元素
        int length = kernelMatrix.length;
        float[] result = new float[length];
        System.arraycopy(kernelMatrix[selectedItemIndex], 0, result, 0, length);
        return result;
    }

    private float[] fetchCiOptimal(int startIndex, int endIndex, int selectedItemIndex, float[][] cis) {
        float[] result = new float[endIndex - startIndex];
        for (int i = startIndex; i < endIndex; i++) {
            result[i] = cis[i][selectedItemIndex];
        }
        return result;
    }

    private float[] fetchDiagonal(float[][] kernelMatrix) {
        float[] diagonal = new float[kernelMatrix.length];
        for (int i = 0; i < kernelMatrix.length; i++) {
            diagonal[i] = kernelMatrix[i][i];
        }
        return diagonal;
    }

    private int fetchDiagonalAndFindMaxIndex(float[] di2s, float[][] kernelMatrix) {
        float maxValue = Float.MIN_VALUE;
        int index = 0;
        for (int i = 0; i < kernelMatrix.length; i++) {
            di2s[i] = kernelMatrix[i][i];
            if (di2s[i] > maxValue) {
                maxValue = di2s[i];
                index = i;
            }
        }
        return index;
    }

    private float[][] calculateSimilarityMatrix(float[][] featureMatrix) throws InterruptedException {
        int row = featureMatrix.length;
        int dimension = featureMatrix[0].length;
        float[][] similarityMatrix = new float[row][row];
        int remainder = row % threadNum;
        int quotient = row / threadNum;
        int segment = (remainder == 0) ? quotient : quotient + 1;
        CountDownLatch countDownLatch = new CountDownLatch(threadNum);
        for (int loop = 0; loop < threadNum; loop++) {
            int finalLoop = loop;
            threadPool.submit(() -> calculateSegmentSimilarityMatrix(segment, finalLoop, row, dimension, similarityMatrix,
                    featureMatrix, countDownLatch));
        }
        countDownLatch.await();
        return similarityMatrix;
    }

    private float[][] calculateKernelMatrix(float[] scores, float[][] similarities) throws InterruptedException {
        int row = similarities.length;
        float[][] kernelMatrix = new float[row][row];
        int remainder = row % threadNum;
        int quotient = row / threadNum;
        int segment = (remainder == 0) ? quotient : quotient + 1;
        CountDownLatch countDownLatch = new CountDownLatch(threadNum);
        //按照线程数 横向切分矩阵
        for (int loop = 0; loop < similarities.length; loop++) {
            int finalLoop = loop;
            threadPool.submit(() -> calculateSegmentKernelMatrix(segment, finalLoop, row, kernelMatrix, scores, similarities,
                    countDownLatch));
        }
        countDownLatch.await(100, TimeUnit.SECONDS);
        return kernelMatrix;
    }

    private void calculateSegmentKernelMatrix(int segment, int loop, int row, float[][] kernelMatrix, float[] scores,
                                              float[][] similarityMatrix, CountDownLatch countDownLatch) {
        for (int rowIndex = segment * loop; rowIndex < segment * (loop + 1) && rowIndex < row; rowIndex++) {
            for (int colIndex = 0; colIndex < row; colIndex++) {
                kernelMatrix[rowIndex][colIndex] = scores[rowIndex] * similarityMatrix[rowIndex][colIndex] * scores[colIndex];
            }
        }
        countDownLatch.countDown();
    }

    private void calculateSegmentSimilarityMatrix(int segment, int loop, int row, int dimension, float[][] similarityMatrix,
                                                  float[][] featureMatrix, CountDownLatch countDownLatch) {
        for (int rowIndex = segment * loop; rowIndex < segment * (loop + 1) && rowIndex < row; rowIndex++) {
            for (int colIndex = 0; colIndex < row; colIndex++) {
                for (int k = 0; k < dimension; k++) {
                    similarityMatrix[rowIndex][colIndex] += featureMatrix[rowIndex][k] * featureMatrix[colIndex][k];
                }
            }
        }
        countDownLatch.countDown();
    }

}
