import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# ====================================================
# 1. 模型架構
# ====================================================
class IDS_Model(nn.Module):

    def __init__(self, input_size):

        super(IDS_Model, self).__init__()

        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 2)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)

        return x


# ====================================================
# 2. 頁面設定
# ====================================================
st.set_page_config(
    page_title="ScamSense Pro",
    layout="wide"
)

st.title("🛡️ IDS 魯棒性深度分析平台")
st.markdown("---")


# ====================================================
# 3. 載入模型與資料
# ====================================================
@st.cache_resource
def load_resources():

    df = pd.read_csv("cleaned_portscan_data.csv")

    # 抽樣測試資料
    test_samples = df.sample(200, random_state=42)

    X_raw = test_samples.drop("Label", axis=1)

    y_raw = torch.LongTensor(
        test_samples["Label"].values
    )

    input_size = X_raw.shape[1]

    # 原始模型
    model_raw = IDS_Model(input_size)

    model_raw.load_state_dict(
        torch.load("ids_model.pth")
    )

    model_raw.eval()

    # 防禦模型
    model_def = IDS_Model(input_size)

    model_def.load_state_dict(
        torch.load("ids_model_defended.pth")
    )

    model_def.eval()

    return X_raw, y_raw, model_raw, model_def


X_df, y, model_raw, model_def = load_resources()


# ====================================================
# 4. FGSM 攻擊
# ====================================================
def run_attack_test(model, data, labels, eps):

    data_tensor = torch.FloatTensor(
        data.values
    ).requires_grad_(True)

    outputs = model(data_tensor)

    loss = nn.CrossEntropyLoss()(outputs, labels)

    model.zero_grad()

    loss.backward()

    # FGSM Attack
    perturbed = (
        data_tensor +
        eps * data_tensor.grad.data.sign()
    )

    # 限制範圍
    perturbed = torch.clamp(
        perturbed,
        0,
        1
    ).detach()

    with torch.no_grad():

        final_outputs = model(perturbed)

        _, pred = torch.max(
            final_outputs,
            1
        )

        acc = (
            (pred == labels)
            .sum()
            .item()
            / labels.size(0)
        )

    return acc, pred


# ====================================================
# 5. Sidebar
# ====================================================
with st.sidebar:

    st.header("⚙️ 實驗參數")

    epsilon = st.slider(
        "對抗性擾動強度 (Epsilon)",
        0.0,
        0.5,
        0.1,
        0.01
    )

    st.markdown("---")

    st.write("📌 指標說明")

    st.write("- ε vs Accuracy")
    st.write("- Confusion Matrix")
    st.write("- Defense Performance")


# ====================================================
# 6. 執行攻擊測試
# ====================================================
acc_raw, pred_raw = run_attack_test(
    model_raw,
    X_df,
    y,
    epsilon
)

acc_def, pred_def = run_attack_test(
    model_def,
    X_df,
    y,
    epsilon
)


# ====================================================
# 7. ε vs Accuracy 趨勢圖
# ====================================================
st.subheader(
    "1️⃣ 攻擊強度 (ε) 與 偵測準確率 關係圖"
)

eps_range = [
    0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5
]

raw_trends = [
    run_attack_test(
        model_raw,
        X_df,
        y,
        e
    )[0] * 100
    for e in eps_range
]

def_trends = [
    run_attack_test(
        model_def,
        X_df,
        y,
        e
    )[0] * 100
    for e in eps_range
]

fig_trend = go.Figure()

# 原始模型
fig_trend.add_trace(
    go.Scatter(
        x=eps_range,
        y=raw_trends,
        name="原始模型",
        line=dict(
            color="#ff4b4b",
            width=3
        )
    )
)

# 防禦模型
fig_trend.add_trace(
    go.Scatter(
        x=eps_range,
        y=def_trends,
        name="防禦模型",
        line=dict(
            color="#28a745",
            width=3
        )
    )
)

fig_trend.update_layout(
    template="plotly_dark",
    xaxis_title="Epsilon",
    yaxis_title="Accuracy (%)",
    height=400
)

st.plotly_chart(
    fig_trend,
    use_container_width=True
)


# ====================================================
# 8. 混淆矩陣
# ====================================================
st.subheader(
    "2️⃣ 決策品質分析：混淆矩陣"
)

# 中間增加空白
col_cm1, spacer, col_cm2 = st.columns([1, 0.3, 1])


def plot_confusion_matrix(
    y_true,
    y_pred,
    title,
    color
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # 🔥 更小尺寸
    fig, ax = plt.subplots(
        figsize=(2.6, 2.3)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap=color,
        ax=ax,
        cbar=False,

        annot_kws={
            "size": 8
        },

        xticklabels=['N', 'A'],
        yticklabels=['N', 'A']
    )

    ax.set_title(
        title,
        fontsize=9
    )

    ax.set_xlabel(
        'Pred',
        fontsize=7
    )

    ax.set_ylabel(
        'True',
        fontsize=7
    )

    ax.tick_params(
        axis='both',
        labelsize=7
    )

    plt.tight_layout(
        pad=0.5
    )

    return fig


# 原始模型矩陣
with col_cm1:

    st.pyplot(
        plot_confusion_matrix(
            y.numpy(),
            pred_raw.numpy(),
            f"Raw (ε={epsilon})",
            "Reds"
        ),
        use_container_width=False
    )


# 防禦模型矩陣
with col_cm2:

    st.pyplot(
        plot_confusion_matrix(
            y.numpy(),
            pred_def.numpy(),
            f"Defense (ε={epsilon})",
            "Greens"
        ),
        use_container_width=False
    )


# ====================================================
# 9. 攻防效能總覽表
# ====================================================
st.subheader(
    "3️⃣ 攻防效能總覽對比表"
)

results_data = {

    "評估場景": [
        "基準測試 (ε=0)",
        "對抗性攻擊",
        "防禦提升"
    ],

    "原始模型": [
        f"{raw_trends[0]:.1f}%",
        f"{acc_raw*100:.1f}%",
        "-"
    ],

    "防禦模型": [
        f"{def_trends[0]:.1f}%",
        f"{acc_def*100:.1f}%",
        f"+{int((acc_def-acc_raw)*100)}%"
    ]
}

st.table(
    pd.DataFrame(results_data)
)


# ====================================================
# Footer
# ====================================================
st.markdown("---")

st.caption(
    "ScamSense Pro | Adversarial Attack Analysis"
)
