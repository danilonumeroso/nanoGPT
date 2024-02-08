import unittest
import torch
from model import SlidingWindowSelfAttention, GPTConfig

class TestModel(unittest.TestCase):
    def setUp(self):
        self.configs = [
            GPTConfig(sliding_window=1),
            GPTConfig(block_size=5, sliding_window=2),
            GPTConfig(block_size=5, sliding_window=3),
            GPTConfig(block_size=16, sliding_window=16),
            GPTConfig(block_size=16, sliding_window=17)
        ]

        self.attn_layers = [SlidingWindowSelfAttention(config) for config in self.configs]

    def test_init(self):
        for attn_layer, config in zip(self.attn_layers, self.configs):
            self.assertEqual(attn_layer.n_head, config.n_head)
            self.assertEqual(attn_layer.n_embd, config.n_embd)
            self.assertEqual(attn_layer.dropout, config.dropout)
            self.assertIsInstance(attn_layer.c_attn, torch.nn.Linear)
            self.assertIsInstance(attn_layer.c_proj, torch.nn.Linear)
            self.assertIsInstance(attn_layer.attn_dropout, torch.nn.Dropout)
            self.assertIsInstance(attn_layer.resid_dropout, torch.nn.Dropout)

    def test_attn_mask(self):
        attn_mask_1 = torch.eye(self.configs[0].block_size)
        attn_mask_2 = torch.tensor([[1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [0, 1, 1, 0, 0],
                                    [0, 0, 1, 1, 0],
                                    [0, 0, 0, 1, 1]])
        attn_mask_3 = torch.tensor([[1, 0, 0, 0, 0],
                                    [1, 1, 0, 0, 0],
                                    [1, 1, 1, 0, 0],
                                    [0, 1, 1, 1, 0],
                                    [0, 0, 1, 1, 1]])
        
        attn_mask_4 = torch.tril(torch.ones(self.configs[3].block_size, self.configs[3].block_size))
        attn_mask_5 = torch.tril(torch.ones(self.configs[4].block_size, self.configs[4].block_size))

        attn_masks = [attn_mask_1, attn_mask_2, attn_mask_3, attn_mask_4, attn_mask_5]

        for attn_layer, attn_mask, config in zip(self.attn_layers, attn_masks, self.configs):
            self.assertTrue(torch.all(attn_layer.attn_mask.eq(attn_mask)))
            self.assertTrue(torch.all(attn_layer.attn_mask.sum(-1) <= torch.fill(torch.zeros(config.block_size), config.sliding_window)))

    def test_c_attn_weights(self):
        for attn_layer, config in zip(self.attn_layers, self.configs):
            self.assertEqual(attn_layer.c_attn.weight.shape, (3 * config.n_embd, config.n_embd))

    def test_c_proj_weights(self):
        for attn_layer, config in zip(self.attn_layers, self.configs):
            self.assertEqual(attn_layer.c_proj.weight.shape, (config.n_embd, config.n_embd))

if __name__ == '__main__':
    unittest.main()