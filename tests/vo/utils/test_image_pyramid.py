import numpy as np

from pyvo.utils import RGBDImagePyramid

from unittest import TestCase
import pytest


class TestImagePyramid(TestCase):

    def setUp(self) -> None:
        self.maxDiff = None

        rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(123456789)))
        self.image = rs.randint(0, 256, size=(160, 160)).astype(np.uint8)  # Random RGB image
        self.depth = rs.randint(0, 65536, size=(160, 160)).astype(np.uint16)  # Random depth image
        self.intrinsics = np.array([
            [517.3, 0.0, 318.6],
            [0.0, 516.5, 239.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def tearDown(self) -> None:
        return super().tearDown()

    def test__given_different_shapes_and_levels__when_build_pyramids__then_raises_value(self):

        # Given
        nb_levels = 3
        pyramid = RGBDImagePyramid(levels=nb_levels)
        diff_depth = np.zeros((500, 500), dtype=np.uint16)

        # When + Then
        with pytest.raises(ValueError):
            pyramid.build_pyramids(
                gray_image=self.image, depth_image=diff_depth, camera_intrinsics=self.intrinsics
            )

    def test__given_a_image_and_levels__when_build_pyramids__then_right_resolution(self):

        # Given
        nb_levels = 3

        # When
        pyramid = RGBDImagePyramid(levels=nb_levels)
        pyramid.build_pyramids(gray_image=self.image, depth_image=self.depth, camera_intrinsics=self.intrinsics)

        # Then
        self.assertEqual(pyramid.gray_at(0).shape, (160, 160))
        self.assertEqual(pyramid.gray_at(1).shape, (80, 80))
        self.assertEqual(pyramid.gray_at(2).shape, (40, 40))

        self.assertEqual(pyramid.depth_at(0).shape, (160, 160))
        self.assertEqual(pyramid.depth_at(1).shape, (80, 80))
        self.assertEqual(pyramid.depth_at(2).shape, (40, 40))

        self.assertEqual(pyramid.intrinsics_at(0).shape, (3, 3))
        self.assertEqual(pyramid.intrinsics_at(1).shape, (3, 3))
        self.assertEqual(pyramid.intrinsics_at(2).shape, (3, 3))

    def test__given_a_pyramid__when_update__then_updates_data(self):

        # Given
        nb_levels = 3

        pyramid_a = RGBDImagePyramid(levels=nb_levels)
        pyramid_b = RGBDImagePyramid(levels=nb_levels)
        pyramid_b.build_pyramids(gray_image=self.image, depth_image=self.depth, camera_intrinsics=self.intrinsics)

        # When
        pyramid_a.update(pyramid_b)

        # Then
        for level in range(nb_levels):
            self.assertEqual(pyramid_a.gray_at(level).shape, pyramid_b.gray_at(level).shape)
            self.assertEqual(pyramid_a.depth_at(level).shape, pyramid_b.depth_at(level).shape)

            np.testing.assert_allclose(
                pyramid_a.gray_at(level), pyramid_b.gray_at(level),
            )
            np.testing.assert_allclose(
                pyramid_a.depth_at(level), pyramid_b.depth_at(level)
            )
            np.testing.assert_allclose(
                pyramid_a.intrinsics_at(level), pyramid_b.intrinsics_at(level)
            )
