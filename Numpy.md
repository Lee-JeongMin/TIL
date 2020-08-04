## Numpy

* ndarray.ndim : 배열의 축(차원)의 수를 반환한다.

```python
a = np.array([[1,2,3], [1,5,9], [3,5,7]])
print(a.ndim)
# 결과 
# 2
```

* ndarray.shape : 배열의 형태를 반환한다. 

```python
a = np.array([[1,2,3], [1,5,9], [3,5,7]])
print(a.shape)
# 결과
# (3,3)
```

* ndarray.size : 배열 내 원소의 총 개수를 반환한다.