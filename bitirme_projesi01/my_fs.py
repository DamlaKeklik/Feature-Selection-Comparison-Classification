import math
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# --- 1. Spearman (threshold-based) ---

def rank(data):
    """Verilen veri listesini sıralayıp ranklarını döndürür (average rank)"""
    sorted_data = sorted((val, idx) for idx, val in enumerate(data))
    ranks = [0] * len(data)
    i = 0
    while i < len(data):
        same_val_indices = [sorted_data[i][1]]
        j = i + 1
        while j < len(data) and sorted_data[j][0] == sorted_data[i][0]:
            same_val_indices.append(sorted_data[j][1])
            j += 1
        avg_rank = sum(range(i + 1, j + 1)) / (j - i)
        for idx in same_val_indices:
            ranks[idx] = avg_rank
        i = j
    return ranks

def spearman_corr(x, y):
    """Spearman sıralama korelasyon katsayısını hesaplar"""
    if len(x) != len(y):
        raise ValueError("Uzunluklar eşit değil")
    n = len(x)
    rx = rank(x)
    ry = rank(y)
    d2 = [(a - b) ** 2 for a, b in zip(rx, ry)]
    return 1 - (6 * sum(d2)) / (n * (n**2 - 1))

def spearman_selector_auto(X, y, threshold=0.2):
    """Spearman korelasyonuna göre threshold üzerindeki özellikleri seçer"""
    correlations = {}
    y_list = y.tolist()
    for col in X.columns:
        x_list = X[col].tolist()
        rho = spearman_corr(x_list, y_list)
        correlations[col] = abs(rho)

    selected_features = [col for col, val in correlations.items() if val >= threshold]
    return X[selected_features], dict((col, correlations[col]) for col in selected_features)

# --- 2. Mutual Information (threshold-based) ---
def entropy(values):
    total = len(values)
    counts = Counter(values)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())

def mutual_information(x, y):
    total = len(x)
    joint_counter = Counter(zip(x, y))
    x_counter = Counter(x)
    y_counter = Counter(y)

    mi = 0.0
    for (xi, yi), joint_count in joint_counter.items():
        p_xy = joint_count / total
        p_x = x_counter[xi] / total
        p_y = y_counter[yi] / total
        mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi

def mutual_info_selector_auto(X, y, threshold=0.01):
    """Mutual Information skoruna göre threshold üzerindeki özellikleri seçer"""
    mi_scores = {}
    y_list = y.tolist()
    for col in X.columns:
        x_list = X[col].tolist()
        mi = mutual_information(x_list, y_list)
        mi_scores[col] = mi

    selected_features = [col for col, val in mi_scores.items() if val >= threshold]
    return X[selected_features], dict((col, mi_scores[col]) for col in selected_features)

# --- 3. Random Forest Mantığıyla Manual (Bilgi Kazancı ile) ---
def split_by_threshold(x, threshold):
    """Bir eşik değere göre x'i ikiye ayırır"""
    left, right = [], []
    for val in x:
        if val <= threshold:
            left.append(val)
        else:
            right.append(val)
    return left, right

def info_gain(x, y, threshold):
    """Verilen eşik değerine göre bilgi kazancı hesaplar"""
    y_left = [yi for xi, yi in zip(x, y) if xi <= threshold]
    y_right = [yi for xi, yi in zip(x, y) if xi > threshold]

    if not y_left or not y_right:
        return 0

    total = len(y)
    h_y = entropy(y)
    h_left = entropy(y_left)
    h_right = entropy(y_right)

    weight_left = len(y_left) / total
    weight_right = len(y_right) / total

    return h_y - (weight_left * h_left + weight_right * h_right)

def best_threshold_info_gain(x, y):
    """Tüm olası eşikler için bilgi kazancını hesaplayıp en iyisini döndürür"""
    sorted_vals = sorted(set(x))
    best_gain = 0
    for i in range(1, len(sorted_vals)):
        threshold = (sorted_vals[i - 1] + sorted_vals[i]) / 2
        gain = info_gain(x, y, threshold)
        if gain > best_gain:
            best_gain = gain
    return best_gain

def rf_selector_auto(X, y, threshold=0.01):
    """Bilgi kazancı skoruna göre threshold üzerindeki özellikleri seçer"""
    scores = {}
    y_list = y.tolist()
    for col in X.columns:
        x_list = X[col].tolist()
        gain = best_threshold_info_gain(x_list, y_list)
        scores[col] = gain

    selected_features = [col for col, val in scores.items() if val >= threshold]
    return X[selected_features], dict((col, scores[col]) for col in selected_features)

def auto_optimize_threshold_selector(selector_func, X, y, model, scoring="f1"):
    """
    Otomatik olarak optimal threshold'u bulur.
    Performans ve efficiency'yi dengeleyerek en iyi sonucu verir.
    
    :param selector_func: spearman_selector_auto, mutual_info_selector_auto, rf_selector_auto
    :param X: Özellikler (preprocessed)
    :param y: Etiket
    :param model: Kullanılacak sklearn modeli
    :param scoring: "accuracy" veya "f1"
    :return: En iyi threshold, en iyi X subset, skor, seçilen özellik sayısı, tüm sonuçlar
    """
    
    # Selector'a göre threshold aralıklarını belirle
    if selector_func.__name__ == 'spearman_selector_auto':
        # Spearman için daha geniş aralık
        base_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.35, 0.4]
    elif selector_func.__name__ == 'mutual_info_selector_auto':
        # Mutual Info için küçük değerler
        base_thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    else:  # rf_selector_auto
        # RF için küçük değerler
        base_thresholds = [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
    
    results = []
    max_features = X.shape[1]
    
    for thresh in base_thresholds:
        try:
            X_sel, scores_dict = selector_func(X, y, threshold=thresh)
            feature_count = X_sel.shape[1]
            
            if feature_count == 0:
                continue  # hiçbir özellik seçilmediyse atla
            
            # Model performansını değerlendir
            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
            clf = model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            if scoring == "accuracy":
                performance = accuracy_score(y_test, y_pred)
            else:
                performance = f1_score(y_test, y_pred)
            
            # Efficiency score hesapla
            feature_ratio = feature_count / max_features
            
            # Multi-objective optimization: 
            # - Yüksek performans (60% ağırlık)
            # - Az özellik sayısı (25% ağırlık) 
            # - Dengeli yaklaşım (15% ağırlık - performans/özellik oranı)
            performance_score = 0.6 * performance
            efficiency_score = 0.25 * (1 - feature_ratio)
            balance_score = 0.15 * (performance / feature_ratio if feature_ratio > 0 else 0)
            
            total_score = performance_score + efficiency_score + balance_score
            
            results.append({
                'threshold': thresh,
                'performance': performance,
                'feature_count': feature_count,
                'feature_ratio': feature_ratio,
                'total_score': total_score,
                'X_selected': X_sel,
                'scores_dict': scores_dict
            })
            
        except Exception as e:
            continue  # Hata durumunda bu threshold'u atla
    
    if not results:
        # Hiç sonuç bulunamazsa, en düşük threshold ile dene
        min_thresh = min(base_thresholds)
        X_sel, scores_dict = selector_func(X, y, threshold=min_thresh)
        if X_sel.shape[1] > 0:
            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
            clf = model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            performance = f1_score(y_test, y_pred) if scoring == "f1" else accuracy_score(y_test, y_pred)
            return min_thresh, X_sel, performance, X_sel.shape[1], [{
                'threshold': min_thresh,
                'performance': performance,
                'feature_count': X_sel.shape[1],
                'total_score': performance
            }]
        else:
            # Son çare: tüm özellikleri kullan
            return 0, X, 0.5, X.shape[1], []
    
    # En iyi sonucu seç
    best_result = max(results, key=lambda x: x['total_score'])
    
    return (best_result['threshold'], 
            best_result['X_selected'], 
            best_result['performance'], 
            best_result['feature_count'],
            results)

def get_comprehensive_thresholds(selector_func, X, y):
    """
    Selector fonksiyonuna göre daha kapsamlı threshold listesi oluşturur
    """
    if selector_func.__name__ == 'spearman_selector_auto':
        return [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 
                0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 
                0.22, 0.25, 0.27, 0.3, 0.32, 0.35, 0.37, 0.4, 0.42, 0.45]
    elif selector_func.__name__ == 'mutual_info_selector_auto':
        return [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 
                0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
                0.012, 0.015, 0.017, 0.02, 0.022, 0.025, 0.027, 0.03, 0.035, 0.04]
    else:  # rf_selector_auto
        return [0.0001, 0.0002, 0.0003, 0.0005, 0.0007, 0.001, 0.0015, 0.002, 
                0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01,
                0.012, 0.015, 0.017, 0.02, 0.022, 0.025, 0.027, 0.03, 0.035, 0.04]

# Legacy functions (backward compatibility)
def efficient_threshold_selector(selector_func, X, y, model, thresholds, scoring="f1", 
                                min_performance=0.70, efficiency_weight=0.3):
    """Legacy function - now uses auto_optimize_threshold_selector"""
    best_thresh, best_X_selected, best_performance, best_feature_count, all_results = auto_optimize_threshold_selector(
        selector_func, X, y, model, scoring
    )
    return best_thresh, best_X_selected, best_performance, best_feature_count

def optimize_threshold_selector(selector_func, X, y, model, thresholds, scoring="accuracy"):
    """Legacy function - now uses auto_optimize_threshold_selector"""
    best_thresh, best_X_selected, best_performance, best_feature_count, all_results = auto_optimize_threshold_selector(
        selector_func, X, y, model, scoring
    )
    return best_thresh, best_X_selected, best_performance